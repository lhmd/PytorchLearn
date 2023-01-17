# 第一种方法
import time

# 网络模型
# 数据(输入，标注)
# 损失函数
# 对上面三个地方调用.cuda()就可以了

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

lhmd = LHMD()
##################################.cuda()调用#########################################################################
if torch.cuda.is_available():
    lhmd = lhmd.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
##################################.cuda()调用#########################################################################
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

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
# start_time = time.time()
for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    # 训练步骤开始
    lhmd.train()# 不一定要写这句话
    for data in train_dataloader:
        imgs, targets = data
##################################.cuda()调用#########################################################################
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = lhmd(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        # 防止打印东西太多
        if total_train_step % 100 == 0:
            # end_time = time.time()
            # print(end_time-start_time)
            print("训练次数：{}，LOSS:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    lhmd.eval()# 不一定要写这句话
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
##################################.cuda()调用#########################################################################
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = lhmd(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss“{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(lhmd, "./model/20_lhmd{}.pth".format(i))
    # torch.save(lhmd.state_dict(), "./model/20_lhmd{}.pth".format(i))

writer.close()