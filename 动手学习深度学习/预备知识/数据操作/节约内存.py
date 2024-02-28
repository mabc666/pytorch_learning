import  torch
# 下面操作展示了每次赋值动作其实都是创建了新引用
x=torch.arange(12)
xpoint = id(x)
x = x + torch.arange(12)
print(id(x) == xpoint)

# 下面操作赋值可以避免创建新引用
z = torch.zeros_like(x)
print('id(z)',id(z))
z[:] = x + torch.arange(12)
print('id(z)',id(z))

# 下面操作也可以避免创建新引用
y = torch.tensor([1,2,3,5])
beforeY = id(y)
y += torch.tensor([4,5,6,7])
print(beforeY == id(y))
