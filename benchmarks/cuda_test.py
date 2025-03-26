import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)

with torch.cuda.amp.autocast():
    result = torch.matmul(x, y)

print(result)
