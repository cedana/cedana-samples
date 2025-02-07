#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset

# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Clean up the distributed process group
def cleanup():
    dist.destroy_process_group()

# Simple training loop
def train(rank, world_size, model, tokenizer, dataset, epochs=3):
    setup(rank, world_size)

    # Prepare the model for DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create a DistributedSampler to handle data partitioning
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(tokenized_dataset, batch_size=16, sampler=sampler)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(rank),
                'attention_mask': batch['attention_mask'].to(rank)
            }
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:  # Log every 10 batches
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

    # Save the model on rank 0
    if rank == 0:
        model_save_path = "fine_tuned_model"
        model.module.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("This script requires at least 2 GPUs to run.")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load a dataset (e.g., IMDB for binary classification)
    dataset = load_dataset("imdb")['train']

    # Launch the training process on multiple GPUs
    torch.multiprocessing.spawn(train, args=(world_size, model, tokenizer, dataset), nprocs=world_size, join=True)
