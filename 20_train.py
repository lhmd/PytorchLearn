# 神经网络模型在model.py里面
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
lhmd = LHMD()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(lhmd.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    # 训练步骤开始
    lhmd.train()# 不一定要写这句话
    for data in train_dataloader:
        imgs, targets = data
        outputs = lhmd(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        # 防止打印东西太多
        if total_train_step % 100 == 0:
            print("训练次数：{}，LOSS:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    lhmd.eval()# 不一定要写这句话
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = lhmd(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss“{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(lhmd, "./model/20_lhmd{}.pth".format(i))
    # torch.save(lhmd.state_dict(), "./model/20_lhmd{}.pth".format(i))

writer.close()