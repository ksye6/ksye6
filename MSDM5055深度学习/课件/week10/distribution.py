import torch
from torch import nn

class Distribution(nn.Module):

    def __init__(self, nvars, name = "Distribution"):
        super(Distribution, self).__init__()
        self.name = name
        self.nvars = nvars

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self, batchSize):
        raise NotImplementedError(str(type(self)))

    def logProbability(self, x):
        raise NotImplementedError(str(type(self)))

    def save(self):
        return self.state_dict()

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict
