#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100000
batch_size = 64
lr = 0.01
input_size = 100
num_classes = 10
num_samples = 10000

# Synthetic data
X = torch.randn(num_samples, input_size)
y = torch.randint(0, num_classes, (num_samples,))
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
