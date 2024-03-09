import numpy as np
import torch
from torch.utils import data


def synthetic_data(w, b, num_example):
    X = torch.normal(0, 1, (num_example, len(w)))
    # 矩阵乘法
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,(y.shape))
    return X,y
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# pytorch 数据读取器
def load_array(data_arrays, batch_size, is_train=True):
    '''构造一个pytorch的数据迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=True)
batch_size = 10
data_iter = load_array((features,labels),batch_size)

# 模型定义
# Linear表示全连接层
from torch import nn
net = nn.Sequential(nn.Linear(2,1))

# 初始化模型
#在使用net之前，我们需要初始化模型参数。 如在线性回归模型中的权重和偏置。 深度学习框架通常有预定义的方法来初始化参数。 在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')