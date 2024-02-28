import torch
import numpy as np

#np 和 tr张量互相转换
x=torch.tensor([1,2,3])
A = x.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 将张量转换成python标量
a = torch.tensor([3.5])
print(type(a.item()))