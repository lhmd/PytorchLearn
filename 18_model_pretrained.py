import torchvision.datasets
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')
# 由于数据量过大所以不要运行本程序下载！！！会直接占用C盘内存！！！
# 下载好的训练好的训练模型名称为vgg16_true，没训练好的是vgg16_false下面将会用这个模型进行改进
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16_false.classifier[6] = nn.Linear(4096, 10)