import torch
from torch import nn


class Sinon(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

sinon = Sinon()
x = torch.tensor(1.0)
output = sinon(x)
print(output)