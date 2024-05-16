import torch
import numpy as np
import torch.nn as nn
from reinforcement import NeuralNetwork
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import time
import cv2

torch.manual_seed(42)
trialTime = 20
maxStep = int(4e3)


def simulate(net, trialTime=20, maxStep=4000):
    env = gym.make('Assault-v4', obs_type="ram")
    maxReward = 0
    for episode in range(trialTime):
        s, info = env.reset()
        totalReward = 0
        for step in range(maxStep):
            with torch.no_grad():
                action = net.action(torch.from_numpy(s).unsqueeze(0)/255)
            if type(action) is torch.Tensor:
                action = action.detach().item()
            s2, r, term, trun, info = env.step(action)
            totalReward += r
            s = s2
            print("episode", episode, ", timeStep", step, ", action", action, ", reward", r)

            if term:
                print("ending episode.")
                break

        if totalReward > maxReward:
            maxReward = totalReward
    env.close()
    return maxReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralNetwork().to(device)
net.load(torch.load("C:/Users/’≈√˙Ë∫/Desktop/reinforcement.pth"))
# params = sum(p.numel() for p in dict.values())

params = list(net.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
print('total number of trainable parameters:', nparams)

maxReward = simulate(net, trialTime=trialTime, maxStep=maxStep)
print('best total reward:', maxReward)
