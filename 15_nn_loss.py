import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss1 = L1Loss(reduction='sum')
result1 = loss1(inputs, targets)
# print(result1)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
# print(result_mse)

# 用于分类器,交叉熵
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
# print(x)
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)