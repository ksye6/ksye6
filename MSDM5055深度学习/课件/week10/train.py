import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt

from gaussian import Gaussian
from rnvp import RNVP
from test_cases import SimpleMLPreshape

batch_size = 200

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=False)

def train(flow, learning_rate=0.5, epochs=10):
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print('total nubmer of trainable parameters:', nparams)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -flow.logProbability(data).mean()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} \tMLE loss: {:.6f}'.format(epoch, loss.detach().item()))

        test_loss = 0
        for data, target in test_loader:
            MLE = -flow.logProbability(data).mean()
            test_loss += MLE.detach().item()

        test_loss /= (len(test_loader.dataset)/batch_size)
        print('\nTest set: Average MLE loss: {:.4f}\n'.format(test_loss))

        samples, prob = flow.sample(16)
        samples = samples.reshape(4, 4, 28, 28).permute([0, 2, 1, 3]).reshape(4*28, 4*28)

        plt.figure()
        plt.imshow(samples.detach().numpy(), cmap="gray")
        plt.savefig("generate.pdf")
        plt.close()


if __name__ == "__main__":
    prior = Gaussian([1, 28, 28])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(28 * 14), torch.ones(28 * 14)])[torch.randperm(28 * 28)].reshape(1, 28, 28)
        else:
            b = 1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [SimpleMLPreshape([28 * 14, 400, 200, 28 * 14], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    sList = [SimpleMLPreshape([28 * 14, 400, 200, 28 * 14], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]

    flow = RNVP(maskList, tList, sList, prior)

    train(flow, learning_rate=0.001, epochs=10)



