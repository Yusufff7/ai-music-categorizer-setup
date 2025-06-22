import torch
import torch.nn as nn
import torch.optim as optim

# GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"Available GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected.")

# Matrix multiplication test on GPU
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
c = torch.matmul(a, b)

print(f"\nMatrix multiplication result:\n{c}")
print(f"Tensor device: {c.device}")

# Define simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy training data
X = torch.randn(100, 100).to(device)
y = torch.randn(100, 10).to(device)

print("\n🏋️ Training test (should use GPU):")
for epoch in range(10):  # Reduce if you just want a short test
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
