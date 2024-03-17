import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data/", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data/", train=False, transform=trans, download=True)
# 迭代器
train_iter = data.DataLoader(mnist_train, batch_size=18,shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=18,shuffle=True)

#初始化模型参数
# 784 -> 256 ->10
num_inputs, num_outputs, num_hiddens = 784,10,256
# w1是784 -> 256全连接的参数的W值一共应该有784 * 256个
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1是加在隐藏层上的只有256个
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# w2是256 -> 10全连接的参数的W值一共应该有256 * 10个
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X@W1+b1)
    return H@W2+b2
#reduction指定损失不进行统计，这里还可以写sum和mean
loss = nn.CrossEntropyLoss(reduction='none')

num_epochs,lr = 10,0.1
updater = torch.optim.SGD(params,lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)