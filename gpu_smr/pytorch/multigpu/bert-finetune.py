#!/usr/bin/env python3

import os
import signal
import sys
import traceback
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Global variable to store pre-opened log file descriptor
_log_fd = None
_log_file_path = None
_process_id = None


# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


# Clean up the distributed process group
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_signal_handlers(rank):
    """Set up signal handlers for child processes only"""
    global _log_fd, _log_file_path, _process_id
    print(f"[DEBUG] Setting up signal handlers for rank {rank}, PID {os.getpid()}", flush=True)
    # Ensure /tmp/log directory exists
    os.makedirs('/tmp/log', exist_ok=True)
    # Pre-open log file to avoid file operations in signal handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _process_id = f"rank_{rank}"
    _log_file_path = f'/tmp/log/sigbus_{_process_id}_{timestamp}.log'
    try:
        _log_fd = os.open(_log_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        # Write initial header
        current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        header = f"SIGBUS log for process {_process_id} (PID: {os.getpid()})\n"
        header += f"Signal mask: {current_mask}\n"
        header += f"SIGBUS blocked: {signal.SIGBUS in current_mask}\n"
        header += f"Started at: {datetime.now()}\n"
        header += "=" * 50 + "\n"
        os.write(_log_fd, header.encode('utf-8'))
        os.fsync(_log_fd)
        print(f"[DEBUG] Log file created: {_log_file_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create log file {_log_file_path}: {e}", flush=True)
        _log_fd = None
        return
    def handle_sigbus(signum, frame):
        """Handle SIGBUS signal - minimal async-signal-safe operations only"""
        try:
            if _log_fd is not None:
                timestamp = datetime.now().strftime('%H:%M:%S')
                msg = f"\n[{timestamp}] *** SIGBUS RECEIVED *** at PID {os.getpid()}\n"
                os.write(_log_fd, msg.encode('utf-8'))
                # Try to get basic stack info
                try:
                    import traceback
                    tb_lines = traceback.format_stack(frame)
                    os.write(_log_fd, b"Stack trace (last 10 frames):\n")
                    for line in tb_lines[-10:]:
                        os.write(_log_fd, line.encode('utf-8'))
                except:
                    os.write(_log_fd, b"Failed to capture traceback\n")
                # Sync to ensure data is written
                os.fsync(_log_fd)
            # Also print to stderr
            print(f"\n*** SIGBUS in process {_process_id} - check {_log_file_path} ***", file=sys.stderr, flush=True)
        except:
            try:
                print(f"SIGBUS in PID {os.getpid()}", file=sys.stderr)
            except:
                pass
        # Force immediate exit
        os._exit(128 + signal.SIGBUS)
    # Only set up SIGBUS handler
    print(f"[DEBUG] Installing SIGBUS handler for rank {rank}", flush=True)
    signal.signal(signal.SIGBUS, handle_sigbus)
    
    # Check if SIGBUS is blocked and try to unblock it
    current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if signal.SIGBUS in current_mask:
        print(f"[WARNING] SIGBUS is blocked! Attempting to unblock...", flush=True)
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGBUS})
            print(f"[DEBUG] SIGBUS unblocked successfully", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to unblock SIGBUS: {e}", flush=True)
    
    print(f"[DEBUG] SIGBUS handler installed for {_process_id}, logging to {_log_file_path}", flush=True)
    print(f"[DEBUG] Ready to receive manual SIGBUS. Send: kill -SIGBUS {os.getpid()}", flush=True)
    
    # Log that setup is complete
    if _log_fd is not None:
        setup_msg = f"Signal handler setup complete at {datetime.now()}\n"
        setup_msg += f"Send 'kill -SIGBUS {os.getpid()}' to test\n"
        setup_msg += "=" * 50 + "\n"
        os.write(_log_fd, setup_msg.encode('utf-8'))
        os.fsync(_log_fd)


# Simple training loop
def train(rank, world_size, model, tokenizer, dataset, epochs=3):
    print(f"[DEBUG] Starting train function for rank {rank}", flush=True)
    
    # Set up signal handlers for this child process only
    
    # Only setup distributed if we have multiple GPUs
    if world_size > 1:
        print(f"[DEBUG] Setting up distributed training for rank {rank}", flush=True)
        setup(rank, world_size)
        setup_signal_handlers(rank)
        # Prepare the model for DDP
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        print(f"[DEBUG] Single GPU mode, rank {rank}", flush=True)
        setup_signal_handlers(rank)
        model = model.to(rank)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
        )

    print(f"[DEBUG] Tokenizing dataset for rank {rank}", flush=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_dataset.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'label']
    )

    # Create sampler - distributed or regular
    if world_size > 1:
        sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    
    dataloader = DataLoader(tokenized_dataset, batch_size=16, sampler=sampler, shuffle=(sampler is None))
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(f"[DEBUG] Starting training loop for rank {rank}", flush=True)
    
    model.train()
    for epoch in range(epochs):
        if sampler and world_size > 1:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(rank),
                'attention_mask': batch['attention_mask'].to(rank),
            }
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:  # Log every 10 batches
                print(
                    f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}',
                    flush=True,
                )

    # Save the model on rank 0
    if rank == 0:
        model_save_path = 'fine_tuned_model'
        if world_size > 1:
            model.module.save_pretrained(model_save_path)
        else:
            model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f'Model saved to {model_save_path}')

    cleanup()


if __name__ == '__main__':
    print("[DEBUG] Starting main process", flush=True)
    
    world_size = torch.cuda.device_count()
    print(f"[DEBUG] Detected {world_size} GPU(s)", flush=True)
    
    # Allow single GPU for testing
    if world_size < 1:
        raise RuntimeError('This script requires at least 1 GPU to run.')

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Load a dataset (e.g., IMDB for binary classification)
    print("[DEBUG] Loading dataset", flush=True)
    dataset = load_dataset('imdb')['train']

    if world_size == 1:
        print("[DEBUG] Single GPU mode - running directly without spawn", flush=True)
        train(0, world_size, model, tokenizer, dataset)
    else:
        print("[DEBUG] Multi-GPU mode - launching multiprocessing spawn", flush=True)
        # Launch the training process on multiple GPUs
        torch.multiprocessing.spawn(
            train,
            args=(world_size, model, tokenizer, dataset),
            nprocs=world_size,
            join=True,
        )
