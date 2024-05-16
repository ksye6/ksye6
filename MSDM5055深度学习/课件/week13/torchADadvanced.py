import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import lammps
from lammps import LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL

import numpy as np


class _LammpsCal(Function):
    @staticmethod
    def forward(ctx, pos, lammpsEnv):
        ctx.lammpsEnv = lammpsEnv
        ctx.save_for_backward(pos)

        potential = []
        for i in range(pos.shape[0]):
            lammpsEnv.scatter_atoms("x", 1,3, pos[i].cpu().numpy().reshape(-1).ctypes)
            lammpsEnv.command('run 0')
            potential.append(lammpsEnv.numpy.extract_compute("thermo_pe",LMP_STYLE_GLOBAL,LAMMPS_INT))

        potential = np.array(potential)
        potential = torch.from_numpy(potential).to(pos)

        return potential

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pos, = ctx.saved_tensors
        lammpsEnv = ctx.lammpsEnv

        force = []
        for i in range(pos.shape[0]):
            lammpsEnv.scatter_atoms("x", 1,3, pos[i].cpu().numpy().reshape(-1).ctypes)
            lammpsEnv.command('run 0')
            force.append(lammpsEnv.gather_atoms("f", 1, 3))

        force = np.array(force)
        grad_input = - torch.from_numpy(force).to(pos) * grad_output.reshape(-1, 1)

        return grad_input, None


lammpsCal = _LammpsCal.apply


class LammpsModule(nn.Module):
    def __init__(self, lammpsCommandPath, feedback=False):
        super(LammpsModule, self).__init__()

        if feedback:
            lmp = lammps.lammps()
        else:
            lmp = lammps.lammps(cmdargs=['-screen', 'none'])

        commands = open(lammpsCommandPath).read().split('\n')
        lmp.commands_list(commands)
        lmp.command("run 0")

        self.lmp = lmp

    def forward(self, input):
        return lammpsCal(input, self.lmp)


from torch.autograd import gradcheck

path = './alaine/in.ADP.txt'

lmp = lammps.lammps()
lmp.commands_list(open(path).read().split('\n'))
lmp.command('run 0')
x = torch.tensor(lmp.gather_atoms('x', 1, 3), dtype=torch.float64)

lmpMod = LammpsModule(path)

xbatch = x.reshape(1, -1).repeat(100, 1)
ubatch = lmpMod(xbatch)

xinput = x.reshape(1, -1)
xinput.requires_grad_()
assert gradcheck(lmpMod, xinput, eps=1e-6, atol=1e-4)

