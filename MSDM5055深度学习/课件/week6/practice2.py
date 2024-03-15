import numpy as np
import matplotlib.pyplot as plt
import torch
print(torch.cuda.is_available())

##############################################################################################################
a=torch.randn(2,3)
a.shape
a.dtype
a.device

a=a.to(torch.float64)
a.dtype

a=torch.randn(2,3,dtype=torch.float64)
a

a=torch.zeros(2,2,dtype=torch.float16)
a

a=torch.ones(2,2,dtype=torch.float16)
a

b=torch.Tensor.new(a) 
b

##########
a=np.array([[1,2],[3,4]])
a

b=torch.from_numpy(a)
b
b.numpy()

##########  viewҪ�������ڴ棬shape����
a=torch.tensor([[1,2,3],[4,5,6]])
b=a.view(3,2)
b

a.t().view(1,6)
b=(a.t().contiguous()).view(1,6)
b

b=a.reshape(3,2)
b

b=(a.t().reshape(1,6))
b

##########
a=torch.tensor([[1,2],[3,4]])
b=torch.tensor([[1,1],[2,2]])
a*b # a * b ��ʾ��ӦԪ�����
torch.matmul(a,b) # torch.matmul(a, b) ��ʾ�������

##########
a=torch.randn(2,3)
a
a.max()
a.sum(dim=0)
a.sum(dim=1)
a.reshape(-1)
a.reshape(1,2,3)

b=torch.randn(1,3)
b
torch.cat((a,b),dim=0)
torch.cat((a,b,b),dim=0)

########## copy: ǳ���� ��Ҫ.clone()
a=torch.tensor([[1,2,3],[4,5,6]])
b=a[0:]

b[0,0]=0
a

a=torch.tensor([[1,2,3],[4,5,6]])
b=a[0:].clone()

b[0,0]=0
a

########## �Զ��󵼹������������� a ���ݶ�
from torch import autograd
a=torch.tensor([[1,2],[3,4]],dtype=torch.float32,requires_grad=True) # ��ζ�� PyTorch ���Զ����ٶ� a �ļ��㣬���ҿ���ͨ������ backward()
                                                                     # ������ʹ洢 a ���ݶ�
y=torch.sum(a)

grads=autograd.grad(outputs=y,inputs=a)[0] # ���� y ���� a ���ݶ�
grads


########## ������һ����״Ϊ (3, 4, 5) ������ x������ʹ�� permute ��������ά����������Ϊ (4, 5, 3)
a = torch.randn(3,4,5)
a
a.permute(1,2,0)
a

########## ������һ����״Ϊ (3, 4) ������ x������ʹ�� transpose ��������ת��Ϊ��״Ϊ (4, 3) ��������
a = torch.randn(3, 4)
a
a.transpose(0, 1)
a

##########
torch.eye(3) # I

a=torch.tensor([1,2,3,4])
torch.diag(a) # ���ڶԽ�

a = torch.randn(4, 4)
a
a.triu() # ������
a.tril() # ������

a.triu(diagonal=2) # ����ƫ��

##############################################################################################################
# nn.Module �Ƕ���������ģ�͵Ļ�����
# nn.Sequential ��һ����˳����֯��ִ��������������
# nn.ModuleList �����ڴ洢���ģ���������

# nn.Module
# nn.Sequential
# nn.ModuleList

import torch.nn as nn
import torch.nn.functional as F

class ConvResidualBlock(nn.Module):
    def __init__(self, channels, channelsOut=None, activation=F.relu):
        super(ConvResidualBlock, self).__init__()
        if channelsOut is None:
            channelsOut = channels
        
        self.activation = activation
        self.convLayers = nn.ModuleList([nn.Conv2d(channels, channelsOut, kernel_size=3, padding=1),
                                         nn.Conv2d(channelsOut, channelsOut, kernel_size=3, padding=1)])
    
    def forward(self,inputs):
        temps = inputs
        temps = self.convLayers[0](temps)
        temps = self.activation(temps)
        temps = self.convLayers[1](temps)
        temps = inputs + temps
        temps = self.activation(temps)
        return temps

class ConvResidualNet(nn.Module):
    def __init__(self, inChannels, outChannels, hiddenChannels, numBlocks=2, activation=F.relu):
        super(ConvResidualBlock, self).__init__()
        self.hiddenChannels = hiddenChannels
        self.inital = nn.Conv2d(in_channel = inChannels, out_channel = outChannels, kernel_size=1, padding=0)
        self.blocks = nn.ModuleList([ConvResidualBlock(channels=hiddenChannels, activation=activation) for _ in range(numBlock)])
        self.final = nn.Conv2d(hiddenChannels, outChannels, kernel_size=1, padding=0)
    
    def forward(self,inputs):
        temps = self.inital(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final(temps)
        return outputs














