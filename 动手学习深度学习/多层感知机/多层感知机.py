#1、多层感知机如果不加激活函数本质上还是仿射模型，与单层的感知机模型其实本质上没有差。
#2、想要发挥多层网络的潜力，我们需要一个额外的关键要素：在每层仿射变换后对每个隐藏单元应用非线性激活函数。
#3、激活函数可以将我们的模型进化成非线性模型，一般我们在每个全连接层以后都加一个激活函数。
#4、常见激活函数
#4.1 ReLU
import torch
from matplotlib import pyplot as plt
x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
# y = torch.relu(x)
# y = torch.sigmoid(x)
y = torch.tanh(x)
# 创建图形对象和坐标系
fig, ax = plt.subplots()
# 绘制线性图
ax.plot(x.detach(), y.detach() , marker='o', linestyle='-')
# 添加标题和标签
ax.set_title('fun')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
plt.show()
