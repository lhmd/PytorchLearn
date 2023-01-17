import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class LHMD(nn.Module):
    def __init__(self):
        super(LHMD, self).__init__()
        # inplace:是否对原来位置的变量直接进行替换，TRUE时直接替换原来的变量
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

lhmd = LHMD()
# output = lhmd(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = lhmd(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()