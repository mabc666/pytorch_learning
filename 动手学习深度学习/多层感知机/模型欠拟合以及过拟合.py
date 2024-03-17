# 模型欠拟合：当我们训练误差和验证误差都严重，但是训练误差和验证误差之间的泛化误差很小，这种现象为欠拟合
# 模型过拟合：当训练误差明显要小于验证误差的时候表明模型有很严重的过拟合现象

# 多项式回归
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

max_degree = 20 # 多项式最大阶数
n_train, n_test = 100, 100 #训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test,1))
print(features.shape)
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
print(poly_features.shape)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]