import torch
import numpy as np
from torch import nn

from gaussian import Gaussian
from rnvp import RNVP

from numpy.testing import assert_array_equal, assert_almost_equal, assert_array_almost_equal


class SimpleMLPreshape(nn.Module):
    def __init__(self, dimsList, activation=None, initMethod=None, name="SimpleMLP"):
        super(SimpleMLPreshape,self).__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(dimsList)-2)]
            activation.append(nn.Tanh())
        assert(len(dimsList) == len(activation)+1)
        layerList = []
        self.name = name
        for no in range(len(activation)):
            layerList.append(nn.Linear(dimsList[no],dimsList[no+1]))
            if initMethod is not None:
                initMethod(layerList[-1].weight.data, layerList[-1].bias.data, no)
            if no == len(activation)-1 and activation[no] is None:
                continue
            layerList.append(activation[no])
        self.layerList = torch.nn.ModuleList(layerList)

    def forward(self,x):
        tmp = x.reshape(x.shape[0], -1)
        for layer in self.layerList:
            tmp = layer(tmp)
        return tmp.reshape(x.shape)


def jacobian(y, x):
    assert y.shape[0] == x.shape[0]
    batchsize = x.shape[0]
    dim = y.shape[1]
    res = torch.zeros(x.shape[0],y.shape[1],x.shape[1]).to(x)
    for i in range(dim):
        res[:,i,:] = torch.autograd.grad(y[:,i],x,grad_outputs=torch.ones(batchsize).to(x),create_graph=True,allow_unused=True)[0].reshape(res[:,i,:].shape)
    return res


def test_gaussian_sample():
    batchSize = 100000
    nvars = [2, 8, 8]
    mean = torch.arange(np.prod(nvars)).reshape(nvars).to(torch.float32)
    logsigma = (torch.arange(np.prod(nvars)) * 0.01).reshape(nvars).to(torch.float32)

    prior = Gaussian(nvars, mu=mean, logsigma=logsigma)
    samples = prior.sample(batchSize)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean, decimal=1)
    assert_almost_equal(torch.log(samples.std(0)).detach().numpy(), logsigma, decimal=1)


def test_rnvp_bijective():
    batchSize = 100
    decimal = 3
    prior = Gaussian([3, 32, 32])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm( 3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None])  for _ in range(4)]
    sList = [SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), nn.ELU()]) for _ in range(4)]
    flow = RNVP(maskList, tList, sList, prior)

    x, p = flow.sample(batchSize)
    z, inversep = flow.inverse(x)
    xz, forwardp = flow.forward(z)
    latentp = flow.prior.logProbability(z)
    zx,inversep2 = flow.inverse(xz)

    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(z.detach().numpy(),zx.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(inversep.detach().numpy(),-forwardp.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(inversep.detach().numpy(),inversep2.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(p.detach().numpy(),(latentp-forwardp).detach().numpy(),decimal=decimal)


def test_rnvp_jacobian():
    batchSize = 100
    decimal = 3

    prior = Gaussian([3, 4, 4])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 4 * 2), torch.ones(3 * 4 * 2)])[torch.randperm( 3 * 4 * 4)].reshape(1, 3, 4, 4)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [SimpleMLPreshape([3 * 4 * 2, 20, 20, 3 * 4 * 2], [nn.ELU(), nn.ELU(), None])  for _ in range(4)]
    sList = [SimpleMLPreshape([3 * 4 * 2, 20, 20, 3 * 4 * 2], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    flow = RNVP(maskList, tList, sList, prior)

    z = flow.prior.sample(batchSize).reshape(batchSize, -1).requires_grad_()

    x, forwardp = flow.forward(z.reshape([batchSize] + flow.prior.nvars))
    x = x.reshape(batchSize, -1)

    jac = jacobian(x, z)

    logJac = torch.det(jac)
    logJacP = torch.exp(forwardp.reshape(-1))

    assert_array_almost_equal(logJac.detach().numpy(), logJacP.detach().numpy(), decimal=decimal)


if __name__ == "__main__":
    #test_gaussian_sample()
    #test_rnvp_bijective()
    test_rnvp_jacobian()