import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}; cuda count: {torch.cuda.device_count()}")

t1 = torch.randint(1, 100, (2,6))
print("before move: ", t1)
t1 = t1.to(device)
print("after move: ", t1)

