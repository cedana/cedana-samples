import torch
import torch.nn as nn
import torch.optim as optim

# Ensure we use CPU
device = torch.device("cpu")

# Dummy dataset: 100 samples of 10 features
X = torch.randn(100, 10, device=device)
y = X.clone()  # identity mapping

# Simple 1-layer model
model = nn.Linear(10, 10).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
