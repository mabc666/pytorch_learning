import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import torchvision
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data/", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data/", train=False, transform=trans, download=True)
# 迭代器
train_iter = data.DataLoader(mnist_train, batch_size=18,shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=18,shuffle=True)
net = nn.Sequential(nn.Flatten(),
                   nn.Linear(784,256),
                   nn.ReLU(),
                   nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()