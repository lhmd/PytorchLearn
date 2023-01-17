import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class LHMD(nn.Module):
    def __init__(self):
        super(LHMD, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output

lhmd = LHMD()

step = 0
for data in dataloader:
    imgs, targets = data
    input = torch.flatten(imgs)
    print(input.shape)
    output = lhmd(input)
    print(output.shape)
    step += 1
