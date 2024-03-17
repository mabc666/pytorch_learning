import torch
from torch import nn

# 交叉熵损失的input输入，要么是C，或者（N,C）。C是种类，N是Minibatch的数量
# 交叉熵损失的target是N，表示每个样本中实际的目标是第几个
input  = torch.randn(3,5,requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
loss = torch.nn.CrossEntropyLoss(reduction='mean')
output = loss(input, target)
print(output)


print(torch.softmax(input,1)[range(len(input)),target])
print(input[range(len(input)),target])

def cross_entropy(y_hat, y):
    # [[0, 1],[0,2]] = [0,0] 和 [0, 2]
    return -torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(input, target))
