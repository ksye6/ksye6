import torch
import numpy as np
import bz2
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

dfile = bz2.BZ2File('C://Users//�����//Desktop//ѧҵ//�ۿƴ�//MSDM5055���ѧϰ//��ҵpart2//xyData.bz2')
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


# Ԥ��ͼƬ
import cv2
import torchvision
imgs = torchvision.utils.make_grid(samples).numpy()
# ���һ��
imgs = imgs * 0.5 + 0.5
# ͨ��ת�õ�����ά��
imgs = imgs.transpose(1, 2, 0)
# RGB2BGR
imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
cv2.imshow('win',imgs)
cv2.waitKey(0)


import torch.nn as nn

# ����������
class Generator(nn.Module): # output_size = (input_size - 1) * stride - 2 * padding + kernel_size
    def __init__(self, num_z, ngf, num_channels):
        super(Generator, self).__init__()
        self.num_z = num_z
        self.save_path = './data/generative/dcgan_netG.pth'
        # ÿһ������֮����BN������׼����������DCGAN���ĵ���Ҫ����
        self.main = nn.Sequential(
            # ������Z��������
            # nn.ConvTranspose2d(num_z, ngf * 8, 4, 1, 0, bias=False),    # �������һ��100x1x1���������, ����ߴ�1024x4x4
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
            # nn.Tanh()   # ���sigmoid�������죬���[-1, 1]
            # # [(nc), 64,  64]
            
            nn.ConvTranspose2d(num_z, ngf * 2, 4, 1, 0, bias=False),  #�������һ��100x1x1���������, ����ߴ�256x4x4
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # [256, 4, 4]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # [128, 8, 8]
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()   # ���sigmoid�������죬���[-1, 1]
            # [1, 16, 16]
        )

    def forward(self, input):
        return self.main(input)

# �б�������
class Discriminator(nn.Module):
    def __init__(self, ndf, num_channels):
        super(Discriminator, self).__init__()
        self.save_path = './data/generative/dcgan_netD.pth'
        self.main = nn.Sequential(
            # # ����[(nc), 64,  64]
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
            # nn.Sigmoid()    # ���Ϊ��ĸ���
            
            # ����[1, 16, 16]   output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [128, 8, 8]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [256, 4,  4]
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()    # ���Ϊ��ĸ���
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


# ��ʼ��Ȩ��
def init_weights(m):
    # print(m)
    classname = m.__class__.__name__
    # �����Ȩ�����þ�ֵΪ0����׼��Ϊ0.02
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # BN��Ȩ�����þ�ֵΪ1����׼��Ϊ0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_time_dif(start_time):
    """��ȡ��ʹ��ʱ��"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# �Կ�ѵ��
def train(netG, netD, trainloader, writer):
    start_time = time.time()
    # ѵ��epochs��
    num_epochs = 24
    # ���Ľ���ѧϰ��
    lr = 0.0002
    # Adam�Ż���betas
    betas = (0.5, 0.999)
 
    # ��ٱ�ǩ
    real_label = 1
    fake_label = 0
 
    # ָ����ʧ�������Ż���
    loss_func = nn.BCELoss()    # �����ƽ����غ���
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
 
    iters = 0   # ��ǰ������
    # �����ʧֵ��������
    best_lossD = float('inf')
    best_lossG = float('inf')
    # ��¼�ϴ�Ŀ��ֵ���ӵ�batch��
    last_improveD = 0
    last_improveG = 0
    most_batch = 10000    # ѵ��ֹͣ����
    flag = False  # ��¼�Ƿ����ֹͣ
 
    for epoch in range(num_epochs):
        # ���ݼ������е�ÿ��batch
        for real in trainloader:
            ############################
            # ѵ���б�����Ŀ��Ϊ���log(D(x)) + log(1 - D(G(z)))
            ############################
            real=real.to(device)
            netD.zero_grad()
            batch_size = real.size(0)
            # ʹ����ʵ����ѵ��D
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float)
            real_labels = real_labels.to(device)
            real_outputs = netD(real).view(-1)
            real_outputs = real_outputs.to(device)
            # ������ʵ��������ʧֵ
            lossD_real = loss_func(real_outputs, real_labels)
            # lossD_real.backward()
            # ��ʵ����Ϊ���ƽ������
            D_x = real_outputs.mean().item()
 
            # ʹ���������ļ�����ѵ��D
            # ����Ǳ������z����׼��̬�ֲ�
            z = torch.randn(batch_size, num_z, 1, 1)
            z = z.to(device)
            # ���������ɼ�����
            fake = netG(z)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float)
            fake_labels = fake_labels.to(device)
            # �����������б���
            fake_outputs = netD(fake.detach()).view(-1)
            # �������������ʧֵ
            lossD_fake = loss_func(fake_outputs, fake_labels)
            # lossD_fake.backward()
            # ������Ϊ���ƽ������
            D_G_z1 = fake_outputs.mean().item()
            # ��ʧ��ͣ������򴫲������ݶ�
            lossD = lossD_real + lossD_fake
            lossD.backward()
            # netDִ�в����Ż�
            optimizerD.step()
 
            ############################
            # ѵ����������Ŀ��Ϊ���log(D(G(z)))
            ############################
            netG.zero_grad()
            # ��ǩ��Ϊ��
            labels = torch.full((batch_size,), real_label, dtype=torch.float)
            labels = labels.to(device)
            # ���ü�����ͨ���б������б����ո�ִ�����Ż�
            outputs = netD(fake).view(-1)
            # ���������ͨ���б�����ʧֵ�������򴫲��ݶ�
            lossG = loss_func(outputs, labels)
            lossG.backward()
            # ������Ϊ���ƽ������
            D_G_z2 = outputs.mean().item()
            # ִ���������Ż�
            optimizerG.step()
 
            # �鿴ѵ����Ч��
            if iters % 100 == 0:
                time_dif = get_time_dif(start_time)
                msg = '[{0}/{1}][{2:>6}],  Loss_D: {3:>5.2f},  Loss_G: {4:>5.2f}, ' \
                      'D(x): {5:>5.2} | {6:>5.2} ,  D(G(z)): {7:>5.2}, Time: {8}'
                print(msg.format(epoch+1, num_epochs, iters, lossD.item(), lossG.item(),
                                 D_x, D_G_z1, D_G_z2, time_dif))
                # ���ӻ�ѵ���ɹ�
                writer.add_scalar("loss/Discriminator", lossD.item(), iters)
                writer.add_scalar("loss/Generator", lossG.item(), iters)
 
                # ���������ʧֵ����ʧ��������
                if lossD.item() < best_lossD:
                    best_lossD = lossD.item()
                    last_improveD = iters
                if lossG.item() < best_lossG:
                    best_lossG = lossG.item()
                    last_improveG = iters
 
            # ���������������������ݽ����ɵļ�����
            if iters % 500 == 0:
                g_path = netG.save_path + str(iters)
                torch.save(netG.state_dict(), g_path)
 
            # ѵ��ֹͣ����
            if iters - last_improveD > most_batch and iters - last_improveG > most_batch:
                print("Training Finished ...")
                torch.save(netG.state_dict(), netG.save_path)
                torch.save(netD.state_dict(), netD.save_path)
                flag = True
                break
 
            iters += 1
        # ֹͣѵ��
        if flag:
            break
 
    if not flag:
        print("Training Finished ...")
        torch.save(netG.state_dict(), netG.save_path)
        torch.save(netD.state_dict(), netD.save_path)


if __name__ == '__main__':
    # ������������ӣ���֤ÿ�ν��һ��
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # ѵ��ͼ�����Ųü��ߴ�
    # image_size = 16
    # ͨ����
    num_channels = 1
    # Ǳ������zά��
    num_z = 100
    # ��������������
    num_generator_feature = 128
    # �б����е�������
    num_disc_feature = 16
 
    # # ���ݼ�ת��
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # # �������ݼ���ImageFolderȡ��Ŀ¼���ļ���Ϊ���ݼ�
    # train_set = datasets.ImageFolder(root=root, transform=transform)
    # # �鿴���ݼ���С
    # print('train_set', len(train_set))
    # # ���������
    # train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # 
    # # ���������������
    # samples, _ = next(iter(train_loader))
    # print(samples.size())
 
    # ��ʼ��������
    netG = Generator(num_z, num_generator_feature, num_channels).to(device)
    # ��ʼ���б���
    netD = Discriminator(num_disc_feature, num_channels).to(device)
    # ��ʼȨ�أ������ĸ����Ľ���
    netG.apply(init_weights)
    netD.apply(init_weights)

    from tensorboardX import SummaryWriter
 
    # ��¼ѵ��ָ�꣬���ӻ�չʾ
    log_path = './data/generative/runs'
    with SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())) as writer:
        # ��ʼ�Կ�ѵ��
        train(netG, netD, train_loader, writer)




import cv2
import torchvision

def visualize(samples, name):
    # Ԥ��ͼƬ
    imgs = torchvision.utils.make_grid(samples)
    # ͨ��ת�õ�����ά��
    imgs = imgs.numpy().transpose(1, 2, 0)
    # ���һ��
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
        # ����������ģ�Ͳ���
        iters = 35000
        save_path = netG.save_path + str(iters)
        if not os.path.exists(save_path):
            break
        netG.load_state_dict(torch.load(save_path))
        # ���������ɼ�ͼƬ
        z = torch.randn(64, 100, 1, 1)
        fake = netG(z).detach()
        visualize(fake, str(iters))


# sum(p.numel() for p in net.values())  netG.state_dict() �鿴����














