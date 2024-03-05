# 线性回归的手动实现
# 线性模型
# y = Xw + b
# X=[x1,x2,x3...]
# w = [w1,w2,w3...]
# b 是标量
# y = w1x1+w2x2+w3x3.. + b

# 损失函数 （loss function）
# 度量模型与真实情况之间的误差
# 线性模型中常见的损失函数-平方误差
# l(i) = 1/2 *（y估计- y)^2
# 在整个样本上的平均误差为L(w,b),在训练时需要寻找最小的L(w,b)

# 解析解
# 线性回归模型可以用一个简单的公式表达，可以使用线性代数方法求得，但是深度学习中的问题一般没有解析解。

# 随机梯度下降
# 虽然很多问题没有解析解，但是我们也要使用一种叫梯度下降的方法来训练模型，它不断的在损失函数递减的方向迭代更新参数来降低误差。
# 如果按照全量数据集计算ls函数的值然后在迭代的话比较耗时，可以每次使用一个小批量进行损失计算迭代，此方法叫小批量随机梯度下降。
# 将样本分成n份，每分固定数量a个，计算损失函数的梯度，然后将梯度乘一个正数k，并将其当前参数中减掉。

# 预测
# 使用训练完的w和b，进行计算就称之为预测。

# 矢量化加速
import torch
import numpy as np
import math
import time
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

n = 10000
a = torch.ones([n])
b = torch.ones([n])
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')

# 正太分布与平方损失
# 此节略过有较多的公式，结论如下
# 在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计

