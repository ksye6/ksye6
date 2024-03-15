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

##########  view要求连续内存，shape不用
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
a*b # a * b 表示对应元素相乘
torch.matmul(a,b) # torch.matmul(a, b) 表示矩阵相乘

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

########## copy: 浅拷贝 需要.clone()
a=torch.tensor([[1,2,3],[4,5,6]])
b=a[0:]

b[0,0]=0
a

a=torch.tensor([[1,2,3],[4,5,6]])
b=a[0:].clone()

b[0,0]=0
a

########## 自动求导功能来计算张量 a 的梯度
from torch import autograd
a=torch.tensor([[1,2],[3,4]],dtype=torch.float32,requires_grad=True) # 意味着 PyTorch 将自动跟踪对 a 的计算，并且可以通过调用 backward()
                                                                     # 来计算和存储 a 的梯度
y=torch.sum(a)

grads=autograd.grad(outputs=y,inputs=a)[0] # 计算 y 关于 a 的梯度
grads


########## 假设有一个形状为 (3, 4, 5) 的张量 x，可以使用 permute 函数将其维度重新排列为 (4, 5, 3)
a = torch.randn(3,4,5)
a
a.permute(1,2,0)
a

########## 假设有一个形状为 (3, 4) 的张量 x，可以使用 transpose 函数将其转置为形状为 (4, 3) 的新张量
a = torch.randn(3, 4)
a
a.transpose(0, 1)
a

##########
torch.eye(3) # I

a=torch.tensor([1,2,3,4])
torch.diag(a) # 放在对角

a = torch.randn(4, 4)
a
a.triu() # 上三角
a.tril() # 下三角

a.triu(diagonal=2) # 往右偏移

##############################################################################################################
# nn.Module 是定义神经网络模型的基础类
# nn.Sequential 是一个按顺序组织和执行网络层的容器类
# nn.ModuleList 是用于存储多个模块的容器类

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














