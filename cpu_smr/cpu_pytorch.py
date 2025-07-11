#!/usr/bin/env python3

import signal
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def get_synthetic_data(batch_size=64):
    x = torch.randn(batch_size, 10)
    y = torch.sum(x, dim=1, keepdim=True) + 0.1 * torch.randn(batch_size, 1)
    return x, y


def train():
    model = LinearModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(10**6):
        x, y = get_synthetic_data()
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}")
            print(f'Loss: {loss.item():.6f}')

            with torch.no_grad():
                weights = model.linear.weight.view(-1)
                bias = model.linear.bias.view(-1)
                print('Model weights:', weights[:5].tolist())
                print('Model bias:   ', bias.tolist())

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.view(-1)
                    print(f'Grad {name}:', grad[:5].tolist())

            for i, group in enumerate(optimizer.param_groups):
                print(f"Optimizer lr: {group['lr']}")


def handle_exit(signum, frame):
    sys.exit(1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    train()
