import torch

t1 = torch.arange(0, 100, 2)
print(f"torch.min: {torch.min(t1)}; max: {torch.max(t1)}; mean: {torch.mean(t1.type(torch.float))}; median: {torch.median(t1)}")
#print(f"t1.min: {t1.type(torch.long).min()}")
print(f"t1.min: {t1.min()}; max: {t1.max()}; mean: {t1.type(torch.float).mean()}; median: {t1.median()}")