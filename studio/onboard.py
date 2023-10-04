import torch
print("\n---- scalar ----")
scalar = torch.tensor(7)
print(scalar)
print(scalar.item())
print(scalar.ndim)
print(scalar.shape)


print("\n---- vector ----")
vector = torch.tensor([1,2,3])
print(vector)
print(vector.ndim)
print(vector.shape)

print("\n---- MATRIX ----")
MATRIX = torch.tensor([[1, 2, 3], [3, 4, 5]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

print("\n---- tensor ----")
tensor = torch.tensor([[[1, 2, 3], [3, 4, 5]]])
print(tensor)
print(tensor.ndim)
print(tensor.shape)

# random tensor
print("\n---- random tensor ----")
random_tensor = torch.rand(3, 4)
print(random_tensor)

# use random tensor for image
print("\n---- image using random tensor ----")
random_image = torch.rand(size=(4, 2, 3))
print(random_image)
print(random_image.ndim)

# use random tensor for image
print("\n---- zeros tensor ----")
zeros = torch.zeros(size=(4, 2, 3))
print(zeros)
#size should be same in order to multiple

multipled = random_image * zeros;
print("\n---- multipled by zeros ----")
print(multipled)
