import torch
import numpy as np
import bz2
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import time
from datetime import timedelta

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

class NeuralNetwork(nn.Module):
    def __init__(self, num_z, ngf, num_channels):
        super(NeuralNetwork, self).__init__()
        self.num_z = num_z
        self.save_path = './data/generative/dcgan_netG.pth'
        self.main = nn.Sequential(
            nn.ConvTranspose2d(num_z, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # [256, 4, 4]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # [128, 8, 8]
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # [1, 16, 16]
        )

    def forward(self, input):
        return self.main(input)

    def sample(self, batchSize):
        return self.main((torch.randn(batchSize, self.num_z, 1, 1)).to(device))


class Discriminator(nn.Module):
    def __init__(self, ndf, num_channels):
        super(Discriminator, self).__init__()
        self.save_path = './data/generative/dcgan_netD.pth'
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [128, 8, 8]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [256, 4,  4]
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(netG, netD, trainloader, writer):
    start_time = time.time()
    num_epochs = 24
    lr = 0.0002
    betas = (0.5, 0.999)
 
    real_label = 1
    fake_label = 0
 
    loss_func = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
 
    iters = 0
    best_lossD = float('inf')
    best_lossG = float('inf')
    last_improveD = 0
    last_improveG = 0
    most_batch = 10000
    flag = False
 
    for epoch in range(num_epochs):
        for real in trainloader:
            real=real.to(device)
            netD.zero_grad()
            batch_size = real.size(0)
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float)
            real_labels = real_labels.to(device)
            real_outputs = netD(real).view(-1)
            real_outputs = real_outputs.to(device)
            lossD_real = loss_func(real_outputs, real_labels)
            D_x = real_outputs.mean().item()
            
            z = torch.randn(batch_size, num_z, 1, 1)
            z = z.to(device)
            fake = netG(z)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float)
            fake_labels = fake_labels.to(device)
            fake_outputs = netD(fake.detach()).view(-1)
            lossD_fake = loss_func(fake_outputs, fake_labels)
            D_G_z1 = fake_outputs.mean().item()
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()
 
            netG.zero_grad()
            labels = torch.full((batch_size,), real_label, dtype=torch.float)
            labels = labels.to(device)
            outputs = netD(fake).view(-1)
            lossG = loss_func(outputs, labels)
            lossG.backward()
            D_G_z2 = outputs.mean().item()
            optimizerG.step()
 
            if iters % 100 == 0:
                time_dif = get_time_dif(start_time)
                msg = '[{0}/{1}][{2:>6}],  Loss_D: {3:>5.2f},  Loss_G: {4:>5.2f}, ' \
                      'D(x): {5:>5.2} | {6:>5.2} ,  D(G(z)): {7:>5.2}, Time: {8}'
                print(msg.format(epoch+1, num_epochs, iters, lossD.item(), lossG.item(),
                                 D_x, D_G_z1, D_G_z2, time_dif))
                # 可视化训练成果
                writer.add_scalar("loss/Discriminator", lossD.item(), iters)
                writer.add_scalar("loss/Generator", lossG.item(), iters)
                
                if lossD.item() < best_lossD:
                    best_lossD = lossD.item()
                    last_improveD = iters
                if lossG.item() < best_lossG:
                    best_lossG = lossG.item()
                    last_improveG = iters
            
            if iters % 500 == 0:
                # g_path = netG.save_path + str(iters)
                g_path = netG.save_path
                torch.save(netG, g_path)
            
            if iters - last_improveD > most_batch and iters - last_improveG > most_batch:
                print("Training Finished ...")
                torch.save(netG, netG.save_path)
                torch.save(netD, netD.save_path)
                flag = True
                break
 
            iters += 1
        
        if flag:
            break
 
    if not flag:
        print("Training Finished ...")
        torch.save(netG, netG.save_path)
        torch.save(netD, netD.save_path)

if __name__ == '__main__':
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_channels = 1
    num_z = 100
    num_generator_feature = 128
    num_disc_feature = 16
    
    netG = NeuralNetwork(num_z, num_generator_feature, num_channels).to(device)
    netD = Discriminator(num_disc_feature, num_channels).to(device)
    
    netG.apply(init_weights)
    netD.apply(init_weights)
    
    from tensorboardX import SummaryWriter
    
    log_path = './data/generative/runs'
    with SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())) as writer:
        train(netG, netD, train_loader, writer)
