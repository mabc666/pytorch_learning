import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt


trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data/", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data/", train=False, transform=trans, download=True)
# 迭代器
train_iter = data.DataLoader(mnist_train, batch_size=18,shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=18,shuffle=True)

c,y = next(iter(train_iter))
print('迭代器x数据维度' + str(c.shape))
print('迭代器y数据维度' + str(y.shape))

flatten = nn.Flatten()
flattenedX = flatten(c)
print('展平后的x数据维度' + str(flattenedX.shape))

net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

# 初始化权重
# nn.init是一个用于初始化参数的类
def init_wights(m):
    # 以下代码仅初始化W，b默认为0
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)

# 每一层使用权重初始化
# Sequential中的apply用于对每层执行某个操作
# 常用于初始化参数或者剪枝
net.apply(init_wights)

# 定义交叉熵损失函数
# pytorh中的交叉熵损失计算的是softmax和交叉熵损失的结合
loss = nn.CrossEntropyLoss()

# SGD随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 5
j = 0
print(len(train_iter))
# 训练步骤
# 1.读取数据,X为[18,1,28,28],Y为[18]表示18个数据的标签
# 2.将X输入到net中进行前向传播
# 3.计算损失
# 4.反向传播计算损失函数的梯度
# 5.模型参数更新
# 6.回到第一步
list=[]
j = 0
for i in range(num_epochs):
    j = 0
    loss_sum = 0
    for x,y in train_iter:
        # 计算损失函数
        forward = net(x)
        l = loss(forward, y)
        # 训练器梯度清0
        trainer.zero_grad()
        # 计算损失梯度
        # 样本的平均损失
        l.mean().backward()
        # 记录损失
        loss_sum += l.mean().item()
        j += 1
        # 更新模型
        trainer.step()
        print(f'\r当前epoch:{i} 进度:{((j / 3334) * 100):.2f}% 平均损失:{l.mean():.2f}' , end='')
    # 计算每个epoch的平均损失
    list.append(loss_sum / j)
print('\r\n')
# 创建图形对象和坐标系
# fig, ax = plt.subplots()
# # 绘制线性图
# ax.plot(range(len(list)), list , marker='o', linestyle='-')
# # 添加标题和标签
# ax.set_title('loss graph')
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss_mean')
# ax.grid(True)
# # 显示图形
# plt.show()


# 进行验证
# 准确率
list_acc = []
for x,y in test_iter:
    y_hat = net(x).argmax(axis=1)
    cmp = torch.eq(y_hat, y)
    # 计算相同元素的数量
    num_equal_elements = cmp.sum().item()
    acc = cmp.sum().item() / len(y_hat)
    list_acc.append(acc)
print(f'准确率: {sum(list_acc) / len(list_acc)}')