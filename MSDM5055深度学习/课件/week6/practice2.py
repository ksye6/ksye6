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

##################################################  Resnet  #######################################################
# nn.Module �Ƕ���������ģ�͵Ļ�����
# nn.Sequential ��һ����˳����֯��ִ��������������
# nn.ModuleList �����ڴ洢���ģ���������

# nn.Module
# nn.Sequential
# nn.ModuleList

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, channels, channelsOut=None, activation=F.relu):
        super(ConvResidualBlock, self).__init__()
        if channelsOut is None:
            channelsOut = channels
        self.activation = activation
        self.convLayers = nn.ModuleList([
            nn.Conv2d(channels, channelsOut, kernel_size=3, padding=1),
            nn.Conv2d(channelsOut, channelsOut, kernel_size=3, padding=1),
        ])

    def forward(self, inputs):
        temps = inputs
        temps = self.convLayers[0](temps)
        temps = self.activation(temps)
        temps = self.convLayers[1](temps)
        temps = inputs + temps
        temps = self.activation(temps)
        return temps


class ConvResidualNet(nn.Module):
    def __init__(self, inChannels, outChannels, hiddenChannels, numBlocks=2, activation=F.relu):
        super(ConvResidualNet, self).__init__()
        self.hiddenChannels = hiddenChannels
        self.initial = nn.Conv2d(
            in_channels=inChannels,
            out_channels=hiddenChannels,
            kernel_size=1,
            padding=0
        )
        self.blocks = nn.ModuleList([
            ConvResidualBlock(
                channels=hiddenChannels,
                activation=activation,
            ) for _ in range(numBlocks)
        ])
        self.final = nn.Conv2d(hiddenChannels, outChannels, kernel_size=1, padding=0)

    def forward(self, inputs):
        temps = self.initial(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final(temps)

        return outputs


if __name__ == "__main__":
    net = ConvResidualNet(2, 90*2, 400, 2)

    inputs = torch.randn(10, 2, 16, 16)
    outputs = net(inputs)


##################################################  Unet  #######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvResidualBlock(nn.Module):
    def __init__(self, channels, channelsOut=None, activation=F.relu):
        super(ConvResidualBlock, self).__init__()
        if channelsOut is None:
            channelsOut = channels
        self.activation = activation
        self.convLayers = nn.ModuleList([
            nn.Conv2d(channels, channelsOut, kernel_size=3, padding=1),
            nn.Conv2d(channelsOut, channelsOut, kernel_size=3, padding=1),
        ])

    def forward(self, inputs):
        temps = inputs
        temps = self.convLayers[0](temps)
        temps = self.activation(temps)
        temps = self.convLayers[1](temps)
        temps = inputs + temps
        temps = self.activation(temps)
        return temps


def doubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, inChannels, intermediaChannels=None, outChannels=None, maxHiddenChannels=512, numBlocks=2, useRes=True):
        super(UNet, self).__init__()
        maxNumBlocks = np.log2(maxHiddenChannels)
        assert maxNumBlocks % 1 == 0 and maxNumBlocks > numBlocks

        self.numBlocks = numBlocks

        if intermediaChannels is not None:
            if useRes:
                self.inital = ConvResidualBlock(inChannels, intermediaChannels)
            else:
                self.inital = doubleConv(inChannels, intermediaChannels)
            inChannels = intermediaChannels
        else:
            self.inital = None

        if useRes:
            downConvLst = [ConvResidualBlock(inChannels, int(2**(maxNumBlocks - numBlocks)))]
            self.downConv = nn.ModuleList(
                downConvLst
                +
                [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks + i)),
                    channelsOut = int(2**(maxNumBlocks - numBlocks + 1 + i))
                )
                for i in range(numBlocks)
            ])
        else:
            downConvLst = [doubleConv(inChannels, int(2**(maxNumBlocks - numBlocks)))]
            self.downConv = nn.ModuleList(
                downConvLst
                +
                [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks + i)),
                    out_channels = int(2**(maxNumBlocks - numBlocks + 1 + i))
                )
                for i in range(numBlocks)
            ])

        self.conv1by1 = nn.ModuleList(
            [nn.Conv2d(int(2**(maxNumBlocks - numBlocks)),
                       int(2**(maxNumBlocks - numBlocks)), 1, padding=0),
            ] + [nn.Sequential(
                nn.Conv2d(int(2**(maxNumBlocks - numBlocks + i + 1)),
                          int(2**(maxNumBlocks - numBlocks + i + 1)), 1, padding=0),
                nn.ReLU(inplace=True))
            for i in range(numBlocks)
            ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpooling = nn.MaxPool2d(2)

        if useRes:
            self.upConv = nn.ModuleList(
                [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks + i) + 2**(maxNumBlocks - numBlocks + i + 1)),
                    channelsOut = int(2**(maxNumBlocks - numBlocks + i))
                )
                for i in range(numBlocks -1, -1, -1)
                ] + [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks)) + inChannels,
                    channelsOut = inChannels)]
            )
        else:
            self.upConv = nn.ModuleList(
                [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks + i) + 2**(maxNumBlocks - numBlocks + i + 1)),
                    out_channels = int(2**(maxNumBlocks - numBlocks + i))
                )
                for i in range(numBlocks -1, -1, -1)
                ] + [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks)) + inChannels,
                    out_channels = inChannels)]
            )

        if outChannels is not None:
            if useRes:
                self.final = nn.Sequential(
                    ConvResidualBlock(inChannels, outChannels),
                    nn.Conv2d(outChannels, outChannels, 1)
                )
            else:
                self.final = nn.Sequential(
                    doubleConv(inChannels, outChannels),
                    nn.Conv2d(outChannels, outChannels, 1)
                )
        else:
            self.final = None


    def forward(self, inputs):

        if self.inital is not None:
            inputs = self.inital(inputs)

        context = [inputs]
        for i, down in enumerate(self.downConv):
            inputs = down(inputs)
            inputs = self.maxpooling(inputs)
            inputs = self.conv1by1[i](inputs)
            context.append(inputs)

        for i, term in enumerate(reversed(context[:-1])):
            inputs = self.upsample(inputs)
            inputs = torch.cat([inputs, term], dim=1)
            inputs = self.upConv[i](inputs)

        if self.final is not None:
            inputs = self.final(inputs)

        return inputs

if __name__ == "__main__":
    net = UNet(2, 32, 90*2, 512, 3, useRes=False)
    inputs = torch.randn(10, 2, 16, 16)
    outputs = net(inputs)
        
























