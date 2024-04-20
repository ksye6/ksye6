import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
module_path = 'C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5055深度学习\\作业'
sys.path.append(module_path)

from cifar10Classification import NeuralNetwork, test_loader, batch_size

net = torch.load('C://Users//张铭韬//Desktop//学业//港科大//MSDM5055深度学习//作业//hw3_para//cifarNet.pth')

params = list(net.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
print('total nubmer of trainable parameters:', nparams)

loss_func = nn.CrossEntropyLoss()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将模型移到相同设备上
net.to(device)

test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    net_out = net(data)
    # sum up batch loss
    test_loss += loss_func(net_out, target).detach().item()
    # get the index of the max log-probability
    pred = net_out.detach().max(1)[1]
    correct += pred.eq(target.detach()).sum()

test_loss /= len(test_loader.dataset)
test_loss *= batch_size
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

print('The final score is {:.4f}'.format(((1e2 * correct / len(test_loader.dataset)) / max(int(nparams // 1e4), 1)).item()))
