import torch
x = torch.tensor([1.0,2.0,3.0,4.0],dtype=torch.float32,requires_grad=True)
y = 2 * torch.dot(x,x)
# 通过反向传播计算梯度
y.backward()
print(x.grad)

# 由于梯度会累计所以每次使用时要清一下
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 非标量变量的反向传播
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)

# 由于在计算u的时候计算过y所以可以在这里直接调用反向传播来求x的梯度
x.grad.zero_()
y.sum().backward()
print(x.grad)

# python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100;
    return c
a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()
print((a.grad == d / a))