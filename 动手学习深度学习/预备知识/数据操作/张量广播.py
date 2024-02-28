import torch
# 不同纬度的张量在每个纬度取最大的值作为依据将其余不够的进行补齐
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a)
print(b)
print(a+b)