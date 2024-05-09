import torch
import numpy as np
import bz2
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

dfile = bz2.BZ2File('C://Users//张铭韬//Desktop//学业//港科大//MSDM5055深度学习//作业part2//xyData.bz2')
data = torch.from_numpy(np.load(dfile)).to(torch.float32)
dfile.close()
batch_size = 64

class XYDataset(Dataset):
    def __init__(self, xydata, transformation=None):
        self.xydata = xydata
        self.transformation = transformation

    def __len__(self):
        return self.xydata.shape[0]

    def __getitem__(self, idx):
        ret = self.xydata[idx, :, :, :]
        if self.transformation:
            ret = self.transformation(ret)

        return ret



trainset = XYDataset(data[:-10000, :, :, :])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)
testset = XYDataset(data[10000:, :, :, :])
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
                                         
samples= next(iter(train_loader))
print(samples)


# 预览图片
import cv2
import torchvision
imgs = torchvision.utils.make_grid(samples).numpy()
# 逆归一化
imgs = imgs * 0.5 + 0.5
# 通道转置到最内维度
imgs = imgs.transpose(1, 2, 0)
# RGB2BGR
imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
cv2.imshow('win',imgs)
cv2.waitKey(0)


import torch.nn as nn

# 生成器网络
class Generator(nn.Module): # output_size = (input_size - 1) * stride - 2 * padding + kernel_size
    def __init__(self, num_z, ngf, num_channels):
        super(Generator, self).__init__()
        self.num_z = num_z
        self.save_path = './data/generative/dcgan_netG.pth'
        # 每一层逆卷积之后都有BN批量标准化函数，是DCGAN论文的主要贡献
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            # nn.ConvTranspose2d(num_z, ngf * 8, 4, 1, 0, bias=False),    # 输入的是一个100x1x1的随机噪声, 输出尺寸1024x4x4
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # [(ngf*8), 4,  4]
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # # [(ngf*4), 8,  8]
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # [(ngf*2), 16,  16]
            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # [(ngf), 32,  32]
            # nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            # nn.Tanh()   # 相比sigmoid收敛更快，输出[-1, 1]
            # # [(nc), 64,  64]
            
            nn.ConvTranspose2d(num_z, ngf * 2, 4, 1, 0, bias=False),  #输入的是一个100x1x1的随机噪声, 输出尺寸256x4x4
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # [256, 4, 4]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # [128, 8, 8]
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()   # 相比sigmoid收敛更快，输出[-1, 1]
            # [1, 16, 16]
        )

    def forward(self, input):
        return self.main(input)

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, ndf, num_channels):
        super(Discriminator, self).__init__()
        self.save_path = './data/generative/dcgan_netD.pth'
        self.main = nn.Sequential(
            # # 输入[(nc), 64,  64]
            # nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # [(ndf), 32,  32]
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # # [(ndf*2), 16,  16]
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # [(ndf*4), 8,  8]
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # [(ndf*8), 4,  4]
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()    # 输出为真的概率
            
            # 输入[1, 16, 16]   output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [128, 8, 8]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [256, 4,  4]
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()    # 输出为真的概率
        )

    def forward(self, input):
        return self.main(input)

# x = torch.rand(1, 100, 1, 1)
# model = Generator(100, 128, 1)
# y = model(x)
# print(y.size())
# 
# x = torch.rand(1, 1, 16, 16)
# model = Discriminator(16, 1)
# y = model(x)
# print(y.size())

import torch.nn as nn
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from datetime import timedelta


# 初始化权重
def init_weights(m):
    # print(m)
    classname = m.__class__.__name__
    # 卷积层权重设置均值为0，标准差为0.02
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # BN层权重设置均值为1，标准差为0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 对抗训练
def train(netG, netD, trainloader, writer):
    start_time = time.time()
    # 训练epochs数
    num_epochs = 24
    # 论文建议学习率
    lr = 0.0002
    # Adam优化器betas
    betas = (0.5, 0.999)
 
    # 真假标签
    real_label = 1
    fake_label = 0
 
    # 指定损失函数和优化器
    loss_func = nn.BCELoss()    # 二进制交叉熵函数
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
 
    iters = 0   # 当前总批次
    # 最佳损失值，正无穷
    best_lossD = float('inf')
    best_lossG = float('inf')
    # 记录上次目标值增加的batch数
    last_improveD = 0
    last_improveG = 0
    most_batch = 10000    # 训练停止条件
    flag = False  # 记录是否可以停止
 
    for epoch in range(num_epochs):
        # 数据加载器中的每个batch
        for real in trainloader:
            ############################
            # 训练判别器，目标为最大化log(D(x)) + log(1 - D(G(z)))
            ############################
            real=real.to(device)
            netD.zero_grad()
            batch_size = real.size(0)
            # 使用真实样本训练D
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float)
            real_labels = real_labels.to(device)
            real_outputs = netD(real).view(-1)
            real_outputs = real_outputs.to(device)
            # 计算真实样本的损失值
            lossD_real = loss_func(real_outputs, real_labels)
            # lossD_real.backward()
            # 真实样本为真的平均概率
            D_x = real_outputs.mean().item()
 
            # 使用生成器的假样本训练D
            # 生成潜在向量z，标准正态分布
            z = torch.randn(batch_size, num_z, 1, 1)
            z = z.to(device)
            # 生成器生成假样本
            fake = netG(z)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float)
            fake_labels = fake_labels.to(device)
            # 假样本经过判别器
            fake_outputs = netD(fake.detach()).view(-1)
            # 计算假样本的损失值
            lossD_fake = loss_func(fake_outputs, fake_labels)
            # lossD_fake.backward()
            # 假样本为真的平均概率
            D_G_z1 = fake_outputs.mean().item()
            # 损失求和，并反向传播计算梯度
            lossD = lossD_real + lossD_fake
            lossD.backward()
            # netD执行参数优化
            optimizerD.step()
 
            ############################
            # 训练生成器，目标为最大化log(D(G(z)))
            ############################
            netG.zero_grad()
            # 标签置为真
            labels = torch.full((batch_size,), real_label, dtype=torch.float)
            labels = labels.to(device)
            # 再让假样本通过判别器，判别器刚刚执行了优化
            outputs = netD(fake).view(-1)
            # 计算假样本通过判别器损失值，并反向传播梯度
            lossG = loss_func(outputs, labels)
            lossG.backward()
            # 假样本为真的平均概率
            D_G_z2 = outputs.mean().item()
            # 执行生成器优化
            optimizerG.step()
 
            # 查看训练的效果
            if iters % 100 == 0:
                time_dif = get_time_dif(start_time)
                msg = '[{0}/{1}][{2:>6}],  Loss_D: {3:>5.2f},  Loss_G: {4:>5.2f}, ' \
                      'D(x): {5:>5.2} | {6:>5.2} ,  D(G(z)): {7:>5.2}, Time: {8}'
                print(msg.format(epoch+1, num_epochs, iters, lossD.item(), lossG.item(),
                                 D_x, D_G_z1, D_G_z2, time_dif))
                # 可视化训练成果
                writer.add_scalar("loss/Discriminator", lossD.item(), iters)
                writer.add_scalar("loss/Generator", lossG.item(), iters)
 
                # 更新最佳损失值和损失降低批次
                if lossD.item() < best_lossD:
                    best_lossD = lossD.item()
                    last_improveD = iters
                if lossG.item() < best_lossG:
                    best_lossG = lossG.item()
                    last_improveG = iters
 
            # 保存生成器参数，可以演进生成的假样本
            if iters % 500 == 0:
                g_path = netG.save_path + str(iters)
                torch.save(netG.state_dict(), g_path)
 
            # 训练停止条件
            if iters - last_improveD > most_batch and iters - last_improveG > most_batch:
                print("Training Finished ...")
                torch.save(netG.state_dict(), netG.save_path)
                torch.save(netD.state_dict(), netD.save_path)
                flag = True
                break
 
            iters += 1
        # 停止训练
        if flag:
            break
 
    if not flag:
        print("Training Finished ...")
        torch.save(netG.state_dict(), netG.save_path)
        torch.save(netD.state_dict(), netD.save_path)


if __name__ == '__main__':
    # 设置随机数种子，保证每次结果一样
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # 训练图像缩放裁剪尺寸
    # image_size = 16
    # 通道数
    num_channels = 1
    # 潜在向量z维度
    num_z = 100
    # 生成器中特征数
    num_generator_feature = 128
    # 判别器中的特征数
    num_disc_feature = 16
 
    # # 数据集转换
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # # 构造数据集，ImageFolder取子目录的文件作为数据集
    # train_set = datasets.ImageFolder(root=root, transform=transform)
    # # 查看数据集大小
    # print('train_set', len(train_set))
    # # 构造加载器
    # train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # 
    # # 加载器输出的张量
    # samples, _ = next(iter(train_loader))
    # print(samples.size())
 
    # 初始化生成器
    netG = Generator(num_z, num_generator_feature, num_channels).to(device)
    # 初始化判别器
    netD = Discriminator(num_disc_feature, num_channels).to(device)
    # 初始权重，按论文给出的建议
    netG.apply(init_weights)
    netD.apply(init_weights)

    from tensorboardX import SummaryWriter
 
    # 记录训练指标，可视化展示
    log_path = './data/generative/runs'
    with SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())) as writer:
        # 开始对抗训练
        train(netG, netD, train_loader, writer)




import cv2
import torchvision

def visualize(samples, name):
    # 预览图片
    imgs = torchvision.utils.make_grid(samples)
    # 通道转置到最内维度
    imgs = imgs.numpy().transpose(1, 2, 0)
    # 逆归一化
    imgs = imgs * 0.5 + 0.5
    # RGB2BGR
    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, imgs)
    cv2.waitKey(0)

if __name__ == '__main__':
    import os
    import torch

    netG = Generator(num_z=100, ngf=128, num_channels=1)
    for i in range(11):
        # 加载生成器模型参数
        iters = 35000
        save_path = netG.save_path + str(iters)
        if not os.path.exists(save_path):
            break
        netG.load_state_dict(torch.load(save_path))
        # 生成器生成假图片
        z = torch.randn(64, 100, 1, 1)
        fake = netG(z).detach()
        visualize(fake, str(iters))


# sum(p.numel() for p in net.values())  netG.state_dict() 查看参数














