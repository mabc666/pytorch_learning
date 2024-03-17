import torch
x = torch.tensor([1.0, 0, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# 按元素进行运算，要求张量形状要一致
print(x+y)
print(x-y)
print(x*y)
print(x/y)
# e的n次方
print(torch.exp(x))

# 张量的连接
k = torch.arange(12,dtype=torch.float32).reshape((3,4))
q = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((k,q),dim=0))
print(torch.cat((k,q),dim=1))

# 张量比较
print(k==q)


# 张量求和
print(k.sum())

# 求最大值下标
print('max')
max = torch.randn(10).reshape(2,5)
print(max)
max = max.argmax(axis = 1)
print(max)
