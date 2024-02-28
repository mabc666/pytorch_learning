import torch

# 创建一个张量tensor
x = torch.arange(12)
print(x)

# 获取张量的shape（沿每个轴的长度）
shape = x.shape
print(shape)

# 获取张量的所有元素数量
count = x.numel()
print(count)

# 使用reshape改变张量的形状，但是不该改变张量的数据
y = x.reshape(3,4)
print(y)

# 创建形状为（2，3，4）的张量
z = torch.zeros((2,3,4))
print(z)

# 获取随机正态分布的张量
r = torch.randn(3, 4)
print(r)

# 手动填写张量
m = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
print(m.shape)
print(m)