import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ѭ������ѵ�����ݼ��Ĵ���
n_epochs = 3
# һ��ѵ������������
batch_size_train = 64
# һ�β��Ե���������
batch_size_test = 1000
# ѧϰ��
learning_rate = 0.01
# ������SGD
momentum = 0.9
# ��¼Ƶ��
log_interval = 10
# �������
random_seed = 1
torch.manual_seed(random_seed)

# batch_size (int): ÿ�����εĴ�С����ÿ�ε����з��ص�������������
# shuffle (bool): �Ƿ���ÿ�� epoch ֮ǰ�������ݡ��������Ϊ True������ÿ�� epoch ֮ǰ�����������ݼ��Ի�ø��õ�ѵ��Ч����
 
# torchvision.datasets.MNIST 
# root���������ݵ�Ŀ¼��
# train�����Ƿ����ص���ѵ������
# downloadΪtrueʱ�������������ݼ���ָ��Ŀ¼�У�����Ѵ����򲻻�����
# transform�ǽ���PILͼƬ������ת����汾ͼƬ��ת��������
 
# transform ������һ�� PyTorch ת������������ͼ��ת��Ϊ������������б�׼�������о�ֵΪ 0.1307����׼��Ϊ 0.3081��
# ����ÿһ��ͼ�����ص�ֵ��ȥ��ֵ�����Ա�׼����ʹ���������Ƶĳ߶ȣ��Ӷ������׵�ѵ��ģ�͡�

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

# ��ʼ����������ģ�ͣ�Net��̳�nn.Module�࣬
class Net(nn.Module):
    def __init__(self):
        # ���һЩģ�ͳ�ʼ���ͱ�Ҫ���ڴ����ȹ�����ȷ��Net����ȷ�̳���nn.Module�����й��ܡ�
        super(Net, self).__init__()
 
        # ������һ����Ϊself.conv1�ľ�����������ͨ����Ϊ1����Ϊ�ǻҶ�ͼ�񣩣����ͨ����Ϊ10������˴�СΪ5x5��
        # ����ζ�Ÿþ���㽫���������һ��5x5�ľ���������������ӳ�䵽10�����ͨ���ϡ�
        # ��������ľ��������˵ÿ������˶��������ʼ��Ȩ�غ�ƫ��������ұȽϺ�����ʲô��˼����һ��ʼ�漴��ֵ��Ȼ����ģ�͵ķ��򴫲������У����ǻᱻ��������С����ʧ��������ʹ�������ܹ���׼ȷ�ؽ���Ԥ����
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
 
        # ͬ���ڶ��д��붨����һ����Ϊself.conv2�ľ�����������ͨ����Ϊ10����Ϊself.conv1�����ͨ����Ϊ10����
        # ���ͨ����Ϊ20������˴�СΪ5x5���þ����Ҳ�������������һ��5x5�ľ���������������ӳ�䵽20�����ͨ���ϡ�
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
 
        # ����Dropout2d�㣬����Ľ��Ϳ��Կ������棬��������������������ݣ���ֹ�����
        self.conv2_drop = nn.Dropout2d()
 
        # �������Բ㣨ȫ���Ӳ㣩����һ��320ά������������������ӳ�䵽һ��50ά�������ռ��С�
        self.fc1 = nn.Linear(320, 50)
        # ͬ�������Բ����50ά������������������ӳ�䵽10ά������ռ��С�
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        # relu��������ִ��ReLU��Rectified Linear Unit�����������С��0�������滻Ϊ0��max_pool2d��������ִ�����ػ���������������庬�����
        # ��������ȶ���������xִ��һ�ξ������������ӳ�䵽10�����ͨ���ϣ�Ȼ��Ծ�������Ľ������2x2�����ػ�������
        # ���ػ������в���Ϊ��input������������kernel_size���ػ��㴰�ڴ�С��stride��������Ĭ��Ϊkernel_size
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # ������ͬ����ͬ����������Dropout2d�㣬��ֹ���ݹ����
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
 
        # ������x����״��(batch_size, num_channels, height, width)�任Ϊ(batch_size, 320)��
        # ����batch_size��ʾ�����������������num_channels��ʾ�������ݵ�ͨ������height��ʾ�������ݵĸ߶ȣ�width��ʾ�������ݵĿ�ȡ�
        # �����-1������ʾ�Զ��ƶϸ�ά���ϵĴ�С����Ϊ���Ը����������ݵĴ�С�Զ�ȷ��batch_size�Ĵ�С��
        # ���⣬320�Ĵ�С��ͨ�������ͳػ�����������õ��ġ����������̿��Կ����档
        x = x.view(-1, 320)
 
        # ������xͨ��ȫ���Ӳ�self.fc1�������Ա任��Ȼ��ʹ��ReLU����������������з����Ա任
        x = F.relu(self.fc1(x))
 
        # �ڴ��ݸ�ȫ���Ӳ�֮ǰ������������x��Ӧ��dropout������
        # ���У�self.training��Net���е�һ������ֵ����������ָʾ��ǰģ���Ƿ���ѵ��ģʽ��
        # ��self.trainingΪTrueʱ��dropout�����������ã�����������Ԫ����������������dropout������
        # �������ڲ��Ի�����ģ�͵�ʱ��dropout�����ͻᱻ�رգ�ģ�ͽ�ʹ��������Ȩ��������Ԥ�⣬�����Ԥ��׼ȷ�ԡ�
        # ������Ĳ���˵�����Կ�����
        x = F.dropout(x, training=self.training)
 
        # ������xͨ��ȫ���Ӳ�self.fc1�������Ա任������ӳ����10��ͨ���У�������Ϊ�����뽫���Ϊ10�����
        x = self.fc2(x)
 
        # ����softmax����Ȼ����ȡ������softmax��������ĺ�����Կ����棬�����֮������ȡÿ�����ĸ��ʲ���һ��
        # dim=1 ��˼�Ƕ�x�ĵڶ�ά�Ƚ��в�����x����Ϊ��batch_size, 10����Ҳ������10��num_classes������ط����в�����
        # ȡ������ԭ��Ҳ��Ϊ�˸�����������ȡ��ʧ��������Կ�����Ĳ�����͡�
        return F.log_softmax(x, dim=1)

# ����������
network = Net()
 
# ʹ��SGD������ݶ��½����Ż�����ע�������SGD�Ǵ������ģ�������Ϳ��Կ����棬����SGD��GD������
# network.parameters()��Ҫѵ���Ĳ�����lr��ѧϰ�ʣ�������֮һ��momentum�Ƕ�����Ҳ�ǳ�����֮һ����������Ķ����������Ϊ0.9
# network.parameters()����һ��������Net�������п�ѵ�������ĵ���������Щ��ѵ�����������������е�Ȩ�غ�ƫ����ȡ�
# ���Ż�����ʹ����������������Ը����Ż�����Ҫ������Щ��������С����ʧ������
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
# ���ڼ�¼�ͻ�ͼ�Ĳ���
train_losses = []
train_counter = []
test_losses = []
# [0, 60000, 120000, 180000]
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
# print(test_counter)
# breakpoint1 = input("print(test_counter): ")  # ����������ʲô


def train(epoch):
    # ���ڽ�����������Ϊѵ��ģʽ����ѵ��������ʱ����Ҫ�����л���ѵ��ģʽ���Ա�����Batch Normalization��Dropout�ȸ߼��Ż�������
    # ������ÿ��batch������Ϻ󣬿�����������м�״̬�����ݶȣ���׼����һ��ѵ����
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
 
        # ���������ģ�Ͳ������ݶ���Ϣ����ѵ�������У��Ż������ۼ�ÿ���������ݶȣ�
        # �����ÿ�� batch �����������Ҫʹ�� zero_grad() ���֮ǰ�ۼƵ��ݶȣ��������һ�� batch �ļ������Ӱ�졣
        optimizer.zero_grad()
 
        # ��ȡѵ�����
        output = network(data)
 
        # ��ʧ�������壬����ʹ�õ��Ǹ�������Ȼ��ʧ������negative log likelihood loss����������Ϳ��Կ�����
        # F.nll_loss()����������ģ�������Ŀ�����֮��Ĳ��죬������һ������ֵ��Ϊ��ʧֵ����ֵԽС��ʾģ��Խ�ӽ�Ŀ�ꡣ
        loss = F.nll_loss(output, target)
 
        # loss.backward() �����������㵱ǰ mini-batch ����ʧ��������ģ�Ͳ������ݶȵĴ��롣
        # �ú������ڼ������ݶ�֮�����Ǵ洢�ڲ����� grad �����С����ţ����ǿ���ʹ�� optimizer.step() ����������ģ�Ͳ�����
        # ����ʹ�õ��Ƿ��򴫲��㷨��backpropagation������������������ݶȣ�
        loss.backward()
 
        # ʹ���Ż�������ģ�Ͳ���
        optimizer.step()
 
        # ÿ��log_interval��batch��ӡһ��ѵ����Ϣ��������ǰepoch��batch�š��Ѵ������������������������ǰbatch����ʧֵ��
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            # ��¼��ʧֵ
            train_losses.append(loss.item())
 
            # ��¼Ŀǰ��ѵ����ɵ���������
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
 
            # ���������Ȩ�غ��Ż�����״̬���浽Ӳ���ϣ�����֮����غ�ʹ�á�
            # network.state_dict() ����һ���ֵ䣬���а���������������в�������Ȩ�غ�ƫ������Լ����Ƕ�Ӧ�����ơ���Щ�������������ָ������״̬��
            # torch.save() �������ֵ���󱣴浽ָ�����ļ�·���ϡ���һ��������Ҫ����Ķ��󣬵ڶ����������ļ�·����
            torch.save(network.state_dict(), './model.pth')
 
            # �����ڱ����Ż�����״̬��
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    # ������������Ϊ����ģʽ��������ģʽ�£�ģ�ͻ�ͣ���ض����裬��Dropout�㡢Batch Normalization��ȣ�
    # ����ʹ��ѵ���ڼ�ѧ���Ĳ���������Ԥ�⣬��������ѵ�����Ͻ����ݶȷ��򴫲���Ȩ�ظ��¡�
    network.eval()
 
    test_loss = 0
    correct = 0
 
    # �ر��ݶȼ��㣬�Լ����ڴ����ĺͼӿ�ģ���������̡�����ζ�ţ���ʹ��torch.no_grad()ʱ��ģ�͵Ĳ������ᱻ���º��Ż���
    # ͬʱ������ͼҲ���ᱻ���ٺͼ�¼����ʹ��ǰ�򴫲����ٶȸ��졣
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # ��¼ģ�͵���ʧֵ��reduction='sum'��ʾ��ÿ����������ʧ��ͺ��ٷ��ء�
            test_loss += F.nll_loss(output, target, reduction='sum').item()
 
            # �ҵ����������������У�ÿһ�У�����1ά�ȣ������Ǹ����Լ������ڵ�λ�ã�����һ��Ԫ�顣����Ԫ��ĵ�һ��Ԫ�ؾ��������Ǹ������ڶ���Ԫ�����������λ�á�
            # ����� keepdim=True ������ʾ����ά�Ȳ��䣬Ҳ����˵���صĽ������ά����ԭʼ������ͬ��
            # [1] ��ʾȡ�����Ԫ��ĵڶ���Ԫ�أ�Ҳ����λ����Ϣ�������˱��� pred����� pred ��������������Ԥ�����������ǩ��
            # ����ά����ԭʼ������ 1 ά����ͬ����ʾÿһ����������������Ӧ������ǩ��
            pred = output.data.max(1, keepdim=True)[1]
 
            # target.data.view_as(pred)�� target �������� pred ��������״�������ܣ�reshape����
            # �����˵�����᷵��һ���� pred ������״��ͬ������������ target ��������������
            # .eq()��������������֮�����Ԫ�رȽϣ��õ�һ���ɲ���ֵ��ɵ���������ʾpred��target.data.view_as(pred)�е�ÿ��Ԫ���Ƿ���ȡ�
            # �����Ԫ����ȣ����Ӧλ��ΪTrue������ΪFalse��
            # .sum()����ǰһ���õ���True/False tensor��������ά����ͣ��õ�Ԥ����ȷ����������
            # +=�������batch��Ԥ����ȷ����������ӵ�֮ǰ�Ѿ�������������У��ۼӵõ��������ݼ���Ԥ����ȷ��������(correct)��
            correct += pred.eq(target.data.view_as(pred)).sum()
    # ����������ƽ����ʧ
    test_loss /= len(test_loader.dataset)
    # ��¼���εĲ��Խ��
    test_losses.append(test_loss)
    # ��ӡ���β��Խ�����������Լ���ƽ����ʧ����ȷԤ����������������Լ��������������Լ���ȷԤ�������ռ�ȡ�
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()  # ������������滭ͼ�ͻᱨ��x and y must be the same size����Ϊtest_counter�а�����ģ��δ��ѵ��ʱ������������ӡ�еġ�0����
 
# ������ʽ��ѵ�������Թ��̣������趨��epoch����ѵ��
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
 
# ��������ѵ���Ͳ��Ե�ͼ�񣬰�����¼ѵ������ʧ�仯�����Ե���ʧ�仯
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


# ----------------------------------------------------------- #
# ����ĳ���ѵ�����������������ѵ�������߿�����δӵ�һ����ѵ����ʱ�����state_dicts�м�������ѵ����
# ��ʼ��һ���µ�������Ż�����
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
# ����������ڲ�״̬���Ż������ڲ�״̬
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
 
# Ϊʲô�ǡ�4����ʼ�أ���Ϊn_epochs=3����������[1, n_epochs + 1)
# ��������5��ѵ��
for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()
 
# ����ѵ��������
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


# if __name__ == "__main__":
#     net = NeuralNetwork()
#     train(net)
#     torch.save(net, 'mnistNet.pth')

















