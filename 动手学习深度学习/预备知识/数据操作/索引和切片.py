import torch
x=torch.arange(12,dtype=torch.float32).reshape((3,4))
# 0是第一个元素，-1最后一个元素
print(x[0])
print(x[-1])
print(x[1:3])

# 将数据写入张量指定位置
x[1,2] = 666
print(x)
print(x[1,2])

# 多元素赋值
x[1:3,1:3] = 333
print(x)