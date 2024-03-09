import random
import torch
import matplotlib.pyplot as plt

# 生成数据集线性模型数据集
# y = wx + b + c（噪声）
# x = [x1, x2]
def synthetic_data(w, b, num_example):
    X = torch.normal(0, 1, (num_example, len(w)))
    # 矩阵乘法
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,(y.shape))
    return X,y

true_w = torch.tensor([2.0,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

plt.plot(features, labels)
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 读取数据集
# yield表示获取一次就返回
# batch_size 表示每次返回多少数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机打乱序号
    random.shuffle(indices)
    # 每次调用就给出10个
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 随机初始化模型参数
w = torch.normal(0,1,(2,1),requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
# 每次计算完梯度将梯度清0
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 计算每个批次的损失
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        # backward()求梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
