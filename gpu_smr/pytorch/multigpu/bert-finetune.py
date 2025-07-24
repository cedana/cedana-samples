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


# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


# Clean up the distributed process group
def cleanup():
    dist.destroy_process_group()


def setup_signal_handlers(rank=None):
    """Set up signal handlers for the process"""
    global _log_fd, _log_file_path
    
    # Ensure /tmp/log directory exists
    os.makedirs('/tmp/log', exist_ok=True)
    
    # Pre-open log file to avoid file operations in signal handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    process_id = f"rank_{rank}" if rank is not None else "main"
    _log_file_path = f'/tmp/log/sigbus_{process_id}_{timestamp}.log'
    
    try:
        _log_fd = os.open(_log_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        # Write initial header
        header = f"SIGBUS log for process {process_id} (PID: {os.getpid()})\n"
        os.write(_log_fd, header.encode('utf-8'))
    except Exception as e:
        print(f"Failed to create log file {_log_file_path}: {e}", flush=True)
        _log_fd = None
    
    def handle_sigbus(signum, frame):
        """Handle SIGBUS signal - minimal async-signal-safe operations only"""
        try:
            # Use os.write for async-signal-safe file writing
            if _log_fd is not None:
                msg = f"\nSIGBUS received at PID {os.getpid()}\n"
                os.write(_log_fd, msg.encode('utf-8'))
                
                # Try to get basic stack info (risky but worth attempting)
                try:
                    import traceback
                    tb_lines = traceback.format_stack(frame)
                    for line in tb_lines[-10:]:  # Last 10 stack frames
                        os.write(_log_fd, line.encode('utf-8'))
                except:
                    os.write(_log_fd, b"Failed to capture traceback\n")
                
                # Sync to ensure data is written
                os.fsync(_log_fd)
            
            # Also try to print to stderr (usually works)
            print(f"\nSIGBUS in process {process_id if rank is not None else 'main'} - check {_log_file_path}", file=sys.stderr, flush=True)
            
        except:
            # If even basic operations fail, try stderr
            try:
                print(f"SIGBUS in PID {os.getpid()}", file=sys.stderr)
            except:
                pass
        
        # Force immediate exit - don't try cleanup
        os._exit(128 + signal.SIGBUS)
    
    def handle_exit(signum, frame):
        """Handle normal exit signals"""
        if _log_fd is not None:
            try:
                msg = f"\nReceived signal {signum}, exiting gracefully\n"
                os.write(_log_fd, msg.encode('utf-8'))
                os.close(_log_fd)
            except:
                pass
        sys.exit(1)
    
    # Set up signal handlers
    signal.signal(signal.SIGBUS, handle_sigbus)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    print(f"Signal handlers set up for {process_id}, logging to {_log_file_path}", flush=True)


# Simple training loop
def train(rank, world_size, model, tokenizer, dataset, epochs=3):
    # Set up signal handlers for this child process
    setup_signal_handlers(rank)
    
    setup(rank, world_size)

    # Prepare the model for DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_dataset.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'label']
    )

    # Create a DistributedSampler to handle data partitioning
    sampler = DistributedSampler(
        tokenized_dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(tokenized_dataset, batch_size=16, sampler=sampler)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
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
        model.module.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f'Model saved to {model_save_path}')

    cleanup()


if __name__ == '__main__':
    # Set up signal handlers for the main process
    setup_signal_handlers()
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError('This script requires at least 2 GPUs to run.')

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Load a dataset (e.g., IMDB for binary classification)
    dataset = load_dataset('imdb')['train']

    # Launch the training process on multiple GPUs
    torch.multiprocessing.spawn(
        train,
        args=(world_size, model, tokenizer, dataset),
        nprocs=world_size,
        join=True,
    )
