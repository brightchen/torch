import torch

t1 = torch.rand(2,3)
t2 = torch.rand(2,3)
print("compare two randoms generated without seed: ", t1==t2)  #false

# create random with seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
t1 = torch.rand(2,3)
t2 = torch.rand(2,3)
print("compare two randoms set seed once: ", t1==t2)  #false

torch.manual_seed(RANDOM_SEED)
t1 = torch.rand(2,3)
torch.manual_seed(RANDOM_SEED)
t2 = torch.rand(2,3)
print("compare two randoms set seed once: ", t1==t2)  #true
