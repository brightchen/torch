import torch

#t1 = torch.arange(0, 100, 2)
#one dimension size should be like (20,)
t1 = torch.randint(0, 1000, (20,))
print("dataset/t1: ", t1)

# min, max, mean, median
print(f"torch.min: {torch.min(t1)}; max: {torch.max(t1)}; mean: {torch.mean(t1.type(torch.float))}; median: {torch.median(t1)}")
print(f"t1.min: {t1.min()}; max: {t1.max()}; mean: {t1.type(torch.float).mean()}; median: {t1.median()}")

# position of min, max, mean, median
print(f"torch.argmin: {torch.argmin(t1)}; argmax: {torch.argmax(t1)};\n argsort: {torch.argsort}")
print(f"t1.argmin: {t1.argmin()}; argmax: {t1.argmax()};\n argsort: {t1.argsort()}")


