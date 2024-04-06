import torch
import numpy as np
from torch import nn
import torch.optim as optim

# To properly using gym, you should run
# !pip install "gymnasium[atari, accept-rom-license]"
import gymnasium as gym


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        '''
        Implement your model
        '''
        raise Exception("No implementation")

    def action(self, state):
        '''
        Implement your model
        This method is a must-have, which generate next move.
        The return must be compatible with gym's cart pole actions.
        '''
        raise Exception("No implementation")
        return action

    def implement_your_method_if_needed(self):
        '''
        Implement your model
        '''
        raise Exception("No implementation")


def train(env, net):
    '''
    Train your model
    '''
    raise Exception("No implementation")


if __name__ == "__main__":
    env = gym.make('Assault-v4', obs_type="ram")
    net = NeuralNetwork()
    print(net)
    train(env, net)
    torch.save(net, 'reinforcement.pth')
