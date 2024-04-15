import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import ssl
# 忽略证书验证
ssl._create_default_https_context = ssl._create_unverified_context


batch_size = 128

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

# 输出过程
def progressing(current, total, msg=None):
    progress = current / total
    bar_length = 20
    filled_length = int(round(bar_length * progress))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    if msg:
        print(f'\r[{bar}] {progress * 100:.1f}% {msg}', end='')
    else:
        print(f'\r[{bar}] {progress * 100:.1f}%', end='')
    if current == total - 1:
        print()

# 3×3卷积层
def conv3x3(inChannels, outChannels, stride=1):
    return nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inChannels, outChannels, stride=1):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = conv3x3(inChannels, outChannels, stride)
        self.bn1 = nn.BatchNorm2d(outChannels) # 批归一化层
        self.conv2 = conv3x3(outChannels, outChannels)
        self.bn2 = nn.BatchNorm2d(outChannels) # 批归一化层
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != self.expansion*outChannels:
            self.shortcut = nn.Sequential(nn.Conv2d(inChannels, self.expansion*outChannels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*outChannels)) # 将输入x调整为与残差块的输出具有相同的维度，以便进行元素级的加法操作
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # self.inChannels = 64
        
        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self.formlayer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self.formlayer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self.formlayer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self.formlayer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes) # FC
        self.inChannels = 2
        self.conv1 = conv3x3(3,2)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = self.formlayer(block, 2, num_blocks[0], stride=1)
        self.layer2 = self.formlayer(block, 4, num_blocks[1], stride=2)
        self.layer3 = self.formlayer(block, 8, num_blocks[2], stride=2)
        self.layer4 = self.formlayer(block, 16, num_blocks[3], stride=2)
        self.linear = nn.Linear(16*block.expansion, num_classes) # FC
    
    def formlayer(self, block, outChannels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inChannels, outChannels, stride))
            self.inChannels = outChannels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(ConvResidualBlock, [2,2,2,2])

# 训练
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progressing(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()

# 测试
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progressing(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        torch.save(state, './ckpt.pth')
        best_acc = acc


best_acc = 0  # Start with 0 accuracy
# Instantiate model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet18().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)


start_epoch = 0
# Run training and testing
for epoch in range(start_epoch, start_epoch+80):
    train(epoch)
    test(epoch)

torch.save(model, 'cifarNet.pth')

params = list(model.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
nparams


