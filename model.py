# 这个文件是为20_train服务的，因为文件名不能出现数字所以不更名

import torch
from torch import nn

# 搭建神经网络
class LHMD(nn.Module):
    def __init__(self):
        super(LHMD, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    lhmd = LHMD()
    input = torch.ones((64, 3, 32, 32))
    output = lhmd(input)
    print(output.shape)