import torch
import numpy as np
from distribution import Distribution

class Gaussian(Distribution):
    def __init__(self, nvars, mu=None, logsigma=None, trainable=True, name="gaussian"):
        super(Gaussian,self).__init__(nvars, name)

        if logsigma is None:
            logsigma = torch.zeros(nvars)
        if mu is None:
            mu = torch.zeros(nvars)

        self.mu = torch.nn.Parameter(mu.to(torch.float32), requires_grad=trainable)
        self.logsigma = torch.nn.Parameter(logsigma.to(torch.float32),requires_grad=trainable)

    def sample(self, batchSize):
        size = [batchSize] + self.nvars
        return (torch.randn(size, dtype=self.logsigma.dtype).to(self.logsigma) * torch.exp(self.logsigma) + self.mu)

    def logProbability(self, z):
        return -0.5 * ((z - self.mu)**2 * torch.exp(-2 * self.logsigma)).reshape(z.shape[0], -1).sum(dim = 1, keepdim=True) - (0.5 * np.prod(self.nvars) * np.log(2. * np.pi) + (self.logsigma).sum())
