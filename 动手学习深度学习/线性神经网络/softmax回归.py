# 首先softmax主要用于分类问题
# 可以将某个向量按照其值的大小进行映射，总和为1
import torch
array = torch.tensor(range(10), dtype=torch.float32)

def softmax(array):
    eArray = torch.exp(array)
    sum = eArray.sum(1,keepdim=True )
    return eArray / sum

print(softmax(array).sum())

# 对数似然和交叉熵损失
# softmax函数给出了一个向量y(预测)，我们可以将其视为对于任意x输入来说每个类的条件概率y = p（y=猫｜x）。假设整个数据集（x,y）有n个样本，对于n个样本我们需要最大化这些条件概率
# 即使用极大似然估计，极大似然估计函数就是将n个样本的条件概率连乘，根据最大似然估计我们要最大化P(Y|X)，相当于最小化对数似然，-log(f(x))，此类问题的

def cross_entropy_loss(yhat, y):
    return -((y * torch.log(yhat)).sum())
