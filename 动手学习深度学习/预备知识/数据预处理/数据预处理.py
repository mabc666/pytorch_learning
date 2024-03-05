import os
import pandas as pd
import torch

# 创建文件夹
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
# 创建csv文件
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NaN,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NaN,106000\n')
    f.write('4,NaN,178100\n')
    f.write('NaN,NaN,140000\n')

# 使用panda操作csv文件
data = pd.read_csv(data_file)
print(data)

# 处理文件中的缺失值
# 哪个纬度不填写表示该纬度全部需要，0:2 表示1和2
inputs, outputs = data.iloc[:,0:1], data.iloc[:, 2]
print(inputs)
# 填充均值
inputs = inputs.fillna(inputs.mean())
print(inputs)
# 将数据转换成pytorch中的张量
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(Y)