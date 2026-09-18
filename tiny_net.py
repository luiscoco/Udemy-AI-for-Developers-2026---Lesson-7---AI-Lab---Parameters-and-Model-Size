import torch
import torch.nn as nn

torch.manual_seed(0)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


model = TinyNet()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

print("\nLayer 1 weights BEFORE training:")
print(model.layer1.weight.data)

# Toy regression task: y = sum of inputs, plus noise.
X = torch.randn(64, 4)
y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(64, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for step in range(20):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    print(f"step {step + 1:2d} | loss = {loss.item():.4f}")

print("\nLayer 1 weights AFTER training:")
print(model.layer1.weight.data)
