import torch
import numpy as np
import bz2
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

dfile = bz2.BZ2File('./xyData.bz2')
data = torch.from_numpy(np.load(dfile)).to(torch.float32)
dfile.close()
batch_size = 100


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
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        '''
        Implement your model
        '''
        raise Exception("No implementation")

    def sample(self, batchSize):
        '''
        Implement your model
        This method is a must-have, which generate samples.
        The return must be the generated [batchSize, 1, 16, 16] array.
        '''
        raise Exception("No implementation")
        return samples

    def implement_your_method_if_needed(self):
        '''
        Implement your model
        '''
        raise Exception("No implementation")


def train(net):
    '''
    Train your model
    '''
    raise Exception("No implementation")

if __name__ == "__main__":
    net = NeuralNetwork()
    print(net)
    train(net)
    torch.save(net, 'generative.pth')
