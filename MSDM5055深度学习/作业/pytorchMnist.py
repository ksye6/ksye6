import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_epochs = 3 # ѵ����ѭ������
batch_size_train = 128 # һ��ѵ������������
batch_size_test = 4000 # һ�β��Ե���������
learning_rate = 0.01 # ѧϰ��
momentum = 0.9 # ����SGD
log_interval = 12 # ��¼���

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

# ��¼��ʧ��
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)] # [0, 60000, 120000, 180000]

# Net (1, 1, 28, 28) -> (1, 10, 24, 24);(1, 10, 12, 12) -> (1, 20, 8, 8);(1, 20, 4, 4) -> 20 * 4 * 4 = 320
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 5��5�����1 1 -> 10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 5��5�����2 10 -> 20
        self.conv2_drop = nn.Dropout2d() # ���������ֹ�����
        self.fc1 = nn.Linear(320, 50) # fc1 320 -> 50
        self.fc2 = nn.Linear(50, 10) # fc2 50 -> 10 �ֳ�ʮ��

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 2��2maxpool + ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # 2��2maxpool + ReLU
        x = x.reshape(-1, 320) # (batch_size, num_channels, height, width) -> (batch_size, 320)
        x = F.relu(self.fc1(x))  # fc1
        x = F.dropout(x, training=self.training) # dropout
        x = self.fc2(x) # fc2 �ֳ�ʮ��
        return F.log_softmax(x, dim=1) # softmax -> log, (batch_size, 10)����ά�Ȳ���

network = Net()
# �Ż���
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Train
def train(epoch):
    # ������������Ϊѵ��ģʽ,�Ա�����Batch Normalization��Dropout�ȸ߼��Ż�����,��ÿ��batch������Ϻ�,������������м�״̬(���ݶ�)��׼����һ��ѵ��
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # ÿ�� batch ���������,���֮ǰ�ۼƵ��ݶ�,�������һ�� batch �ļ������Ӱ��.
        optimizer.zero_grad()
        # ��ȡѵ�����
        output = network(data)
        # negative log likelihood loss
        loss = F.nll_loss(output, target)
        # backpropagation
        loss.backward()
        # �Ż�������ģ�Ͳ���
        optimizer.step()
        # ÿ��log_interval��batch��ӡһ��ѵ����Ϣ,������ǰepoch��batch�š��Ѵ������������������������ǰbatch����ʧֵ��
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader),loss.item()))
            # ��¼��ʧֵ
            train_losses.append(loss.item())
            # ��¼Ŀǰ��ѵ����ɵ���������
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # ���������Ȩ�غ��Ż�����״̬���浽Ӳ����,����֮����غ�ʹ��.
            # network.state_dict() ����һ���ֵ�,���а���������������в���(��Ȩ�غ�ƫ����),�Լ����Ƕ�Ӧ������, ���������ָ������״̬.
            torch.save(network.state_dict(), './model.pth')
            # �����Ż�����״̬.
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    # ������������Ϊ����ģʽ.������ģʽ��,ģ�ͻ�ͣ���ض�����,��Dropout�㡢Batch Normalization���,
    # ����ʹ��ѵ���ڼ�ѧ���Ĳ���������Ԥ��,��������ѵ�����Ͻ����ݶȷ��򴫲���Ȩ�ظ���.
    network.eval()
    test_loss = 0
    correct = 0
    # �ر��ݶȼ���,�Լ����ڴ����ĺͼӿ�ģ����������.��ʹ��torch.no_grad()ʱ,ģ�͵Ĳ������ᱻ���º��Ż�
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # ��¼��ʧֵ.reduction='sum'��ʾ��ÿ����������ʧ��ͺ��ٷ���.
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    # ƽ����ʧ
    test_loss /= len(test_loader.dataset)
    # ��¼���εĲ��Խ��
    test_losses.append(test_loss)
    # ��ӡ���β��Խ��,�������Լ���ƽ����ʧ����ȷԤ����������������Լ��������������Լ���ȷԤ�������ռ��.
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

test()

# main
for epoch in range(1, n_epochs + 8):
    train(epoch)
    test()

if __name__ == "__main__":
    torch.save(network, 'mnistNet.pth')






