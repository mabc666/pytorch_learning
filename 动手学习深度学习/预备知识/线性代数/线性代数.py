import torch

# 标量
x = torch.tensor(3.0,dtype=torch.float32)
# 向量
y = torch.arange(12,dtype=torch.float32)
# 张量
A = torch.arange(20,dtype=torch.float32).reshape(5,4)


# 张量算法基本性质
# 深拷贝
B = A.clone()
print(A)
print(A+B)
# A * B 对应元素相乘，这个乘法叫Hadamard积
print(A * B)

# 标量与向量做运算即标量与每个元素做运算
A = A + 1
print(A)
print(2*A)

# 降纬
# 求和
print(A.sum())
print(A)
# 沿着0轴求和，即0轴消失
print(A.sum(axis=0))
# 求和
print(A.mean())
print(A.sum() / A.numel())

# 非降纬求和
print(A.sum(axis=0,keepdim=True))
print(A.sum(axis=0,keepdim=True) / A)
# 按照轴累计求和
print(A)
print(A.cumsum(axis=0))

# 向量点积操作对应位置相乘在相加
one = torch.ones(12,dtype=torch.float32)
print(one)
zero = torch.zeros(12,dtype=torch.float32)
print(one.dot(zero))
print((one * zero).sum())

# 矩阵乘法
X1 = torch.arange(12,dtype=torch.float32).reshape(3,4)
X2 = torch.arange(4,dtype=torch.float32)
X3 = torch.arange(12,dtype=torch.float32).reshape(4,3)
print(X1)
print(X2)
print(torch.mv(X1,X2))
print(torch.mm(X1,X3))

#范数
u = torch.tensor([3.0,-4.0])
#2范数，平方求和在开方
print(u.norm())
#1范数，绝对值求和
print(torch.abs(u).sum())