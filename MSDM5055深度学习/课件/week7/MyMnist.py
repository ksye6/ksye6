import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 循环整个训练数据集的次数
n_epochs = 3
# 一次训练的样本数量
batch_size_train = 64
# 一次测试的样本数量
batch_size_test = 1000
# 学习率
learning_rate = 0.01
# 动量，SGD
momentum = 0.9
# 记录频率
log_interval = 10
# 随机种子
random_seed = 1
torch.manual_seed(random_seed)

# batch_size (int): 每个批次的大小，即每次迭代中返回的数据样本数。
# shuffle (bool): 是否在每个 epoch 之前打乱数据。如果设置为 True，则在每个 epoch 之前重新排列数据集以获得更好的训练效果。
 
# torchvision.datasets.MNIST 
# root：下载数据的目录；
# train决定是否下载的是训练集；
# download为true时会主动下载数据集到指定目录中，如果已存在则不会下载
# transform是接收PIL图片并返回转换后版本图片的转换函数，
 
# transform 函数是一个 PyTorch 转换操作，它将图像转换为张量并对其进行标准化，其中均值为 0.1307，标准差为 0.3081。
# 即将每一个图像像素的值减去均值，除以标准差，如此使它们有相似的尺度，从而更容易地训练模型。

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

# 开始建立神经网络模型，Net类继承nn.Module类，
class Net(nn.Module):
    def __init__(self):
        # 完成一些模型初始化和必要的内存分配等工作。确保Net类正确继承了nn.Module的所有功能。
        super(Net, self).__init__()
 
        # 定义了一个名为self.conv1的卷积层对象，输入通道数为1（因为是灰度图像），输出通道数为10，卷积核大小为5x5。
        # 这意味着该卷积层将对输入进行一次5x5的卷积操作，并将结果映射到10个输出通道上。
        # 不过这里的卷积操作有说每个卷积核都会随机初始化权重和偏置项。这里我比较好奇是什么意思，是一开始随即赋值，然后在模型的反向传播过程中，它们会被更新以最小化损失函数，并使神经网络能够更准确地进行预测吗
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
 
        # 同理，第二行代码定义了一个名为self.conv2的卷积层对象，输入通道数为10（因为self.conv1的输出通道数为10），
        # 输出通道数为20，卷积核大小为5x5。该卷积层也将对其输入进行一次5x5的卷积操作，并将结果映射到20个输出通道上。
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
 
        # 设置Dropout2d层，这个的解释可以看最上面，作用是随机丢弃部分数据，防止过拟合
        self.conv2_drop = nn.Dropout2d()
 
        # 创建线性层（全连接层）接收一个320维的输入张量，并将其映射到一个50维的特征空间中。
        self.fc1 = nn.Linear(320, 50)
        # 同理，此线性层接收50维的特征向量，并将其映射到10维的输出空间中。
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        # relu函数用于执行ReLU（Rectified Linear Unit）激活函数，将小于0的数据替换为0。max_pool2d函数用于执行最大池化层采样操作，具体含义见上
        # 这里就是先对输入数据x执行一次卷积操作，将其映射到10个输出通道上，然后对卷积操作的结果进行2x2的最大池化操作。
        # 最大池化操作中参数为：input，输入张量；kernel_size，池化层窗口大小，stride：步幅，默认为kernel_size
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # 和上面同理，不同的是增加了Dropout2d层，防止数据过拟合
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
 
        # 将输入x的形状从(batch_size, num_channels, height, width)变换为(batch_size, 320)，
        # 其中batch_size表示输入的数据样本数，num_channels表示输入数据的通道数，height表示输入数据的高度，width表示输入数据的宽度。
        # 在这里，-1参数表示自动推断该维度上的大小，因为可以根据输入数据的大小自动确定batch_size的大小。
        # 另外，320的大小是通过卷积层和池化层的输出计算得到的。具体计算过程可以看上面。
        x = x.view(-1, 320)
 
        # 将输入x通过全连接层self.fc1进行线性变换，然后使用ReLU激活函数对输出结果进行非线性变换
        x = F.relu(self.fc1(x))
 
        # 在传递给全连接层之前，在输入张量x上应用dropout操作。
        # 其中，self.training是Net类中的一个布尔值参数，用于指示当前模型是否处于训练模式。
        # 当self.training为True时，dropout操作将被启用，否则所有神经元都被保留，不进行dropout操作。
        # 这样，在测试或评估模型的时候，dropout操作就会被关闭，模型将使用完整的权重来进行预测，以提高预测准确性。
        # 对于其的补充说明可以看上面
        x = F.dropout(x, training=self.training)
 
        # 将输入x通过全连接层self.fc1进行线性变换，最终映射至10个通道中，这是因为我们想将其分为10个类别
        x = self.fc2(x)
 
        # 进行softmax操作然后再取对数，softmax这个操作的含义可以看上面，简而言之就是求取每个类别的概率并归一化
        # dim=1 意思是对x的第二维度进行操作，x现在为（batch_size, 10），也就是在10（num_classes）这个地方进行操作。
        # 取对数的原因也是为了更方便计算和求取损失，具体可以看上面的补充解释。
        return F.log_softmax(x, dim=1)

# 创建神经网络
network = Net()
 
# 使用SGD（随机梯度下降）优化器，注意这里的SGD是带动量的，具体解释可以看上面，还有SGD和GD的区别
# network.parameters()是要训练的参数，lr是学习率，超参数之一；momentum是动量，也是超参数之一。不过这里的动量最好设置为0.9
# network.parameters()返回一个包含了Net类中所有可训练参数的迭代器。这些可训练参数包括神经网络中的权重和偏置项等。
# 在优化器中使用这个迭代器，可以告诉优化器需要更新哪些参数以最小化损失函数。
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
# 用于记录和绘图的参数
train_losses = []
train_counter = []
test_losses = []
# [0, 60000, 120000, 180000]
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
# print(test_counter)
# breakpoint1 = input("print(test_counter): ")  # 看看这里是什么


def train(epoch):
    # 用于将神经网络设置为训练模式。在训练神经网络时，需要将其切换到训练模式，以便启用Batch Normalization和Dropout等高级优化技术，
    # 并且在每个batch处理完毕后，可以清除所有中间状态（如梯度）以准备下一次训练。
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
 
        # 是用于清除模型参数的梯度信息。在训练过程中，优化器会累加每个参数的梯度，
        # 因此在每个 batch 计算结束后，需要使用 zero_grad() 清空之前累计的梯度，避免对下一个 batch 的计算造成影响。
        optimizer.zero_grad()
 
        # 获取训练结果
        output = network(data)
 
        # 损失函数定义，这里使用的是负对数似然损失函数（negative log likelihood loss），具体解释可以看上面
        # F.nll_loss()函数计算了模型输出和目标输出之间的差异，并返回一个标量值作为损失值，该值越小表示模型越接近目标。
        loss = F.nll_loss(output, target)
 
        # loss.backward() 就是用来计算当前 mini-batch 的损失函数关于模型参数的梯度的代码。
        # 该函数会在计算完梯度之后将它们存储在参数的 grad 属性中。接着，我们可以使用 optimizer.step() 函数来更新模型参数。
        # 这里使用的是反向传播算法（backpropagation）来计算网络参数的梯度，
        loss.backward()
 
        # 使用优化器更新模型参数
        optimizer.step()
 
        # 每隔log_interval个batch打印一次训练信息，包括当前epoch、batch号、已处理的样本数、总样本数、当前batch的损失值等
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            # 记录损失值
            train_losses.append(loss.item())
 
            # 记录目前已训练完成的样本数量
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
 
            # 将神经网络的权重和优化器的状态保存到硬盘上，方便之后加载和使用。
            # network.state_dict() 返回一个字典，其中包含了神经网络的所有参数（即权重和偏置项），以及它们对应的名称。这些参数可以用来恢复网络的状态。
            # torch.save() 函数将字典对象保存到指定的文件路径上。第一个参数是要保存的对象，第二个参数是文件路径。
            torch.save(network.state_dict(), './model.pth')
 
            # 这是在保存优化器的状态。
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    # 将神经网络设置为评估模式。在评估模式下，模型会停用特定步骤，如Dropout层、Batch Normalization层等，
    # 并且使用训练期间学到的参数来生成预测，而不是在训练集上进行梯度反向传播和权重更新。
    network.eval()
 
    test_loss = 0
    correct = 0
 
    # 关闭梯度计算，以减少内存消耗和加快模型评估过程。这意味着，在使用torch.no_grad()时，模型的参数不会被更新和优化；
    # 同时，计算图也不会被跟踪和记录，这使得前向传播的速度更快。
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # 记录模型的损失值。reduction='sum'表示将每个样本的损失求和后再返回。
            test_loss += F.nll_loss(output, target, reduction='sum').item()
 
            # 找到输出结果（张量）中，每一行（即第1维度）最大的那个数以及它所在的位置，返回一个元组。其中元组的第一个元素就是最大的那个数，第二个元素是最大数的位置。
            # 这里的 keepdim=True 参数表示保持维度不变，也就是说返回的结果张量维度与原始张量相同。
            # [1] 表示取出这个元组的第二个元素，也就是位置信息，赋给了变量 pred。这个 pred 变量就是神经网络预测出来的类别标签，
            # 它的维度与原始张量第 1 维度相同，表示每一个输入数据样本对应的类别标签。
            pred = output.data.max(1, keepdim=True)[1]
 
            # target.data.view_as(pred)将 target 张量按照 pred 张量的形状进行重塑（reshape）。
            # 具体地说，它会返回一个和 pred 张量形状相同、但数据来自 target 张量的新张量。
            # .eq()方法来进行张量之间的逐元素比较，得到一个由布尔值组成的张量，表示pred和target.data.view_as(pred)中的每个元素是否相等。
            # 如果该元素相等，则对应位置为True，否则为False。
            # .sum()：对前一步得到的True/False tensor沿着所有维度求和，得到预测正确的样本数。
            # +=：将这个batch中预测正确的样本数添加到之前已经处理过的样本中，累加得到整个数据集中预测正确的样本数(correct)。
            correct += pred.eq(target.data.view_as(pred)).sum()
    # 计算样本的平均损失
    test_loss /= len(test_loader.dataset)
    # 记录本次的测试结果
    test_losses.append(test_loss)
    # 打印本次测试结果，包括测试集的平均损失、正确预测的样本数量、测试集的总样本数量以及正确预测的样本占比。
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()  # 不加这个，后面画图就会报错：x and y must be the same size，因为test_counter中包含了模型未经训练时的情况（上面打印中的“0”）
 
# 进入正式的训练、测试过程，根据设定的epoch进行训练
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
 
# 绘制整个训练和测试的图像，包括记录训练的损失变化、测试的损失变化
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


# ----------------------------------------------------------- #
# 检查点的持续训练，继续对网络进行训练，或者看看如何从第一次培训运行时保存的state_dicts中继续进行训练。
# 初始化一组新的网络和优化器。
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
# 加载网络的内部状态、优化器的内部状态
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
 
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
# 继续进行5次训练
for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()
 
# 绘制训练的曲线
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

















