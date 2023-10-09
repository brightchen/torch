
import torch
#NOTES: should not name this file as numpy.py, otherwise "import numpy" actually import from this file
#And cause circular import issue
import numpy as np

#torch.from_numpy
#torch.Tensor.numpy

array = np.arange(1.0, 5.0)
t = torch.from_numpy(array);
print(f"convert numpy to tensor:  numpy array: {array}; tensor: {t}");

#change the array, +3 for each element, the tensor create by the original array not change
array = array + 3
print(f"changed the numpy:  numpy array: {array}; tensor: {t}");

#convert tensor to numpy
array = t.numpy()
print(f"convert tensor to numpy:  numpy array: {array}; tensor: {t}");
#change the tensor, changed tensor also the numpy which created
t += 5
print(f"changed tensor:  numpy array: {array}; tensor: {t}");
t[0] = 100;
print(f"changed tensor:  numpy array: {array}; tensor: {t}");
print(f"dtype:  numpy array: {array.dtype}; tensor: {t.dtype}");






