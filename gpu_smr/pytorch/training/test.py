#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
batch_size = 64
lr = 0.01

# Dummy dataset (MNIST)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True
)

# Simple model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
