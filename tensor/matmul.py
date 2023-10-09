import torch
import time

#%%time
tensor = torch.arange(10000)

#use matmul do the calculation
start_time = time.time()
output = torch.matmul(tensor, tensor)
end_time = time.time()
spent_time = (end_time - start_time)
print(f"matmul spent time: {spent_time}, putput: {output}")

#use @ operator: @ operator is faster than matmul
start_time = time.time()
output = tensor @ tensor
end_time = time.time()
spent_time = (end_time - start_time)
print(f"operator @ spent time: {spent_time}, putput: {output}")


#do the calculation manually
start_time = time.time()
prod = 0;
for i in range(len(tensor)):
  prod += tensor[i] * tensor[i]
end_time = time.time()
print(f"calculate prod manually. spent time: {end_time-start_time}, prod: {prod}")

## conclusion: use matmul is much faster than manually compute prod when data is larger

## matmul dimensions
t1 = torch.rand(2,3)
t2 = torch.rand(3,2)
t3 = t1 @ t2
print(f"t1 shape: {t1.shape}; t2 shape: {t2.shape}; t1@t2 shape: {t3.shape}")

t1 = torch.rand(4,6)
t2 = torch.transpose(t1, 0, 1)
t3 = t1 @ t2
print(f"t1 shape: {t1.shape}; t2 shape: {t2.shape}; t1@t2 shape: {t3.shape}")

