import torch

#torch.reshape;  torch.TreeView;  torch.view_copy;  torch.view_as_XXX
#torch.vstack;  torch.hstack;  torch.stack
#torch.squeeze;  torch.unsqueeze
#torch.permute

t1 = torch.randint(0, 100, (12,))
print("dataset: ", t1)

#reshape
print("t1.reshape((3,4)): \n", t1.reshape((3,4)))
print("t1.reshape((2,6)): \n", t1.reshape((2,6)))
print("t1.reshape((1,12)): \n", t1.reshape((1,12)))
print("t1.reshape((12,1)): \n", t1.reshape((2,3,2)))

#stack
t1 = torch.randint(0, 100, (5,))
t2 = torch.randint(0, 100, (5,))
t3 = torch.randint(0, 100, (5,))
#print("torch.vstack(t1, t2, t3): \n", torch.hstack(t1, t2, t3))

#squeeze: remove useless dimension
t1 = torch.randint(0, 100, (2, 1, 2))
t1_squeezed = torch.squeeze(t1)
print("t1: ", t1)
print(f"t1_squeezed: {t1_squeezed};\n t1_squeezed shape: {t1_squeezed.shape}")
unsqueezed = t1_squeezed.unsqueeze(0)
print(f"unsqueeze(0): {unsqueezed};\n unsqueezed shape: {unsqueezed.shape}")
unsqueezed = t1_squeezed.unsqueeze(1)
print(f"unsqueeze(1): {unsqueezed};\n unsqueezed shape: {unsqueezed.shape}")
unsqueezed = t1_squeezed.unsqueeze(2)
print(f"unsqueeze(2): {unsqueezed};\n unsqueezed shape: {unsqueezed.shape}")

#permute
t0 = torch.arange(0,24)
t1 = torch.reshape(t0, (2, 3, 4))
#t1 = torch.randint(0, 100, (2, 3, 4))

#move the last dimension to first dimension, and become (4, 2, 3)
permuted = t1.permute(2, 0, 1)  #shape (4,2,3)
print("------------------------\norigin: ", t1)
print(f"permute(2, 0, 1): {permuted}; permuted sharp: {permuted.shape}")

#permuted is the view of originally
print(t1[0,0,0]==permuted[0,0,0])
t1[0,0,0] = 100
print(t1[0,0,0]==permuted[0,0,0])
t1[0,0,0] = 0

#select with index
print("------------------------\norigin: ", t1)
print(f"t1[:,0,0]: {t1[:,0,0]}")
print(f"t1[0,:,0]: {t1[0,:,0]}")
print(f"t1[0,0,:]: {t1[0,0,:]}")

t1 = torch.arange(0, 6)
t1 = torch.reshape(t1, (2,3))
print("------------------------\norigin: ", t1)
print(f"t1[:,0]: {t1[:,0]}")
print(f"t1[0,:]: {t1[0,:]}")

