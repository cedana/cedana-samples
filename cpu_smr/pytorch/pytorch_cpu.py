import torch
import torch.nn as nn
import torch.optim as optim
import time

# Ensure we use CPU
device = torch.device("cpu")

# Dummy dataset: 100,000 samples of 100 features
X = torch.randn(100000, 100, device=device)
y = X.clone()  # identity mapping

# Simple 1-layer model (100 -> 100)
model = nn.Linear(100, 100).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    time.sleep(0.5)
