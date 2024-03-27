import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_epochs = 3 # 训练集循环次数
batch_size_train = 128 # 一次训练的样本数量
batch_size_test = 4000 # 一次测试的样本数量
learning_rate = 0.01 # 学习率
momentum = 0.9 # 动量SGD
log_interval = 12 # 记录间隔

random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size_test, shuffle=False)

# 记录损失率
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)] # [0, 60000, 120000, 180000]

# Net (1, 1, 28, 28) -> (1, 10, 24, 24);(1, 10, 12, 12) -> (1, 20, 8, 8);(1, 20, 4, 4) -> 20 * 4 * 4 = 320
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 5乘5卷积层1 1 -> 10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 5乘5卷积层2 10 -> 20
        self.conv2_drop = nn.Dropout2d() # 随机丢弃防止过拟合
        self.fc1 = nn.Linear(320, 50) # fc1 320 -> 50
        self.fc2 = nn.Linear(50, 10) # fc2 50 -> 10 分成十类

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 2乘2maxpool + ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # 2乘2maxpool + ReLU
        x = x.reshape(-1, 320) # (batch_size, num_channels, height, width) -> (batch_size, 320)
        x = F.relu(self.fc1(x))  # fc1
        x = F.dropout(x, training=self.training) # dropout
        x = self.fc2(x) # fc2 分成十类
        return F.log_softmax(x, dim=1) # softmax -> log, (batch_size, 10)对列维度操作

network = Net()
# 优化器
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Train
def train(epoch):
    # 将神经网络设置为训练模式,以便启用Batch Normalization和Dropout等高级优化技术,在每个batch处理完毕后,可以清除所有中间状态(如梯度)以准备下一次训练
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 每个 batch 计算结束后,清空之前累计的梯度,避免对下一个 batch 的计算造成影响.
        optimizer.zero_grad()
        # 获取训练结果
        output = network(data)
        # negative log likelihood loss
        loss = F.nll_loss(output, target)
        # backpropagation
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()
        # 每隔log_interval个batch打印一次训练信息,包括当前epoch、batch号、已处理的样本数、总样本数、当前batch的损失值等
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader),loss.item()))
            # 记录损失值
            train_losses.append(loss.item())
            # 记录目前已训练完成的样本数量
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # 将神经网络的权重和优化器的状态保存到硬盘上,方便之后加载和使用.
            # network.state_dict() 返回一个字典,其中包含了神经网络的所有参数(即权重和偏置项),以及它们对应的名称, 可以用来恢复网络的状态.
            torch.save(network.state_dict(), './model.pth')
            # 保存优化器的状态.
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    # 将神经网络设置为评估模式.在评估模式下,模型会停用特定步骤,如Dropout层、Batch Normalization层等,
    # 并且使用训练期间学到的参数来生成预测,而不是在训练集上进行梯度反向传播和权重更新.
    network.eval()
    test_loss = 0
    correct = 0
    # 关闭梯度计算,以减少内存消耗和加快模型评估过程.在使用torch.no_grad()时,模型的参数不会被更新和优化
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # 记录损失值.reduction='sum'表示将每个样本的损失求和后再返回.
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    # 平均损失
    test_loss /= len(test_loader.dataset)
    # 记录本次的测试结果
    test_losses.append(test_loss)
    # 打印本次测试结果,包括测试集的平均损失、正确预测的样本数量、测试集的总样本数量以及正确预测的样本占比.
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

test()

# main
for epoch in range(1, n_epochs + 8):
    train(epoch)
    test()

if __name__ == "__main__":
    torch.save(network, 'mnistNet.pth')






