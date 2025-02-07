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

    # Create a DistributedSampler to handle data partitioning
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(rank) for k, v in inputs.items()}
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load a dataset (e.g., IMDB for binary classification)
    dataset = load_dataset("imdb")['train']

    # Launch the training process on multiple GPUs
    torch.multiprocessing.spawn(train, args=(world_size, model, tokenizer, dataset), nprocs=world_size, join=True)
