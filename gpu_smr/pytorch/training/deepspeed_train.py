#!/usr/bin/env python3

import torch
import torch.nn as nn
import deepspeed
import argparse

# 1. Define the Model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Argument Parser for DeepSpeed
def add_deepspeed_args(parser):
    """Adds common arguments for the script."""
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', type=int, default=500000, help='number of training epochs')
    
    return parser

# 3. Training Function
def main():
    # --- Setup ---
    parser = argparse.ArgumentParser(description="DeepSpeed Sample Training")
    parser = add_deepspeed_args(parser)
    # Let deepspeed handle its own args
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- DeepSpeed Initialization ---
    deepspeed.init_distributed()

    # Model parameters
    input_size = 1024
    hidden_size = 512
    output_size = 1
    
    model = SimpleModel(input_size, hidden_size, output_size)

    # --- Create a dummy dataset ---
    X_train = torch.randn(args.batch_size * 10, input_size).cuda()
    y_train = torch.randn(args.batch_size * 10, output_size).cuda()
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    
    # --- DeepSpeed Configuration Dictionary ---
    ds_config = {
        "train_batch_size": args.batch_size,
        "optimizer": {
            "type": "Adam",
            "params": { "lr": 0.001 }
        },
        "fp16": { "enabled": True },
        "zero_optimization": { "stage": 2 }
    }

    # Initialize the DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config # Pass the config dictionary here
    )

    criterion = nn.MSELoss()

    # --- Training Loop ---
    print("🚀 Starting DeepSpeed Training...")
    for epoch in range(args.epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(model_engine.device).half()
            labels = labels.to(model_engine.device).half()

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()
        
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print("✅ Training finished.")


if __name__ == '__main__':
    main()
