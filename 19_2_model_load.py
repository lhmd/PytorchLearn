import torch
import torchvision
from torch import nn

# 与前面保存的方式一一对应

# 方式1,加载模型
model1 = torch.load("./model/vgg16_method1.pth")
# print(model1)
# 方拾2，加载模型
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth"))

# 方式1陷阱
# 需要写以下内容，不能直接load
class LHMD(nn.Module):
    def __init__(self):
        super(LHMD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

# 以下内容是不用写的
# lhmd = LHMD()

model = torch.load("./model/lhmd_method1.pth")
print(model)