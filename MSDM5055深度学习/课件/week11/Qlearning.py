import torch
from torch import nn
import numpy as np
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'terminate', 'next'))

class ReplayBuff:
    def __init__(self, maxSize):
        self.buffer = deque()
        self.size = 0
        self.maxSize = maxSize
    def add(self, s, a, r, t, s2):
        if self.size <= self.maxSize:
            self.buffer.append((s,a,r,t,s2))
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s,a,r,t,s2))
    def sample(self, batchSize):
        batch = []
        if self.size < batchSize:
            batch = random.sample(self.buffer, self.size)
        else:
            batch = random.sample(self.buffer, batchSize)

        batch = Transition(*zip(*batch))
        return batch.state, batch.action, batch.reward, batch.terminate, batch.next
    def clear(self):
        self.buffer.clear()
        self.size = 0


class Qnetwork(nn.Module):
    def __init__(self, stateDim, actionDim, tau):
        super(Qnetwork, self).__init__()
        self.linearNN = nn.Sequential(
            nn.Linear(stateDim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, actionDim),
        )
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.tau = tau

    def pred(self, state):
        return self.linearNN(state)