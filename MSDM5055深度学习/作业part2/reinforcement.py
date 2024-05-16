import torch
import numpy as np
import torch.optim as optim

# To properly using gym, you should run
# !pip install "gymnasium[atari, accept-rom-license]"
import gymnasium as gym

import random
import time
from collections import deque

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os

MODELS_ROOT = 'C:/Users/’≈√˙Ë∫/Œƒµµ/data/reinforcement'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linearNN = nn.Sequential(
             nn.Conv2d(1, 256, kernel_size=(5, 5)),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Conv2d(256, 128, kernel_size=(3, 3)),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Conv2d(128, 64, kernel_size=(3, 3)),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Conv2d(64, 32, kernel_size=(3, 3)),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Flatten(),
             nn.Linear(32 * 14 * 14, 7)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        self.epsilon = 0.5
        self.discount = 0.99
        self.min_replay_memory_size = 80000
        self.mini_batch_size = 64
        # self.model = self.linearNN().to(device)
        self.replay_memory = deque(maxlen=100000)
        self.policy_update_counter = 0
    
    def forward(self, input):
        return self.linearNN(input)
    
    def update_replay_memory(self, previous_state, action, reward, current_state, done):
        self.replay_memory.append([previous_state, action, reward, current_state, done])
    
    def predict(self, state):
        self.epsilon = max(0.05, self.epsilon)
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        _, index = torch.max(self.get_q_output(state), 0)
        return index
    
    def get_q_output(self, state):
        with torch.no_grad():
            prediction = self.linearNN(state)
        return prediction[0]
    
    def update_policy(self):
        self.epsilon -= 1 / 50000
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        self.policy_update_counter += 1
        if self.policy_update_counter % 16 != 0:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.mini_batch_size)

        # Get previous states from minibatch, then calculate Q values
        previous_states = torch.Tensor(self.mini_batch_size, 1, 256, 256).to(device)
        torch.cat([mem_item[0] for mem_item in minibatch], dim=0, out=previous_states)

        previous_states /= 255
        with torch.no_grad():
            previous_q_values = self.linearNN(previous_states)
            current_states = torch.Tensor(self.mini_batch_size, 1, 256, 256).to(device)
            torch.cat([mem_item[3] for mem_item in minibatch], dim=0, out=current_states)
            current_states /= 255
            current_q_values = self.linearNN(current_states)

        X = []
        y = []

        for index, (previous_state, action, reward, current_state, done) in enumerate(minibatch):
            if not done:
                max_current_q, _ = torch.max(current_q_values[index], 0)
                new_q = reward + self.discount * max_current_q
            else:
                new_q = reward

            previous_qs = previous_q_values[index]
            previous_qs[action] = new_q

            X.append(previous_state)
            y.append(previous_qs.unsqueeze(0))

        X = torch.cat(X, 0).to(device) / 255
        y = torch.cat(y, 0).to(device)
        self.optimizer.zero_grad()
        outputs = self.linearNN(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
    
    def observation_to_state(self,observation):
        state = cv2.resize(observation, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        state = np.reshape(state, (1, *state.shape))
        return torch.from_numpy(state).float()
    
    def load(self, net):
        self.load_state_dict(net)
    
    def action(self, state):
        tent1 = state.squeeze().numpy()
        tent2 = self.observation_to_state(tent1).to(device).unsqueeze(1)
        action = self.predict(tent2)
        return action


def train(no_episodes=5, save=True, buffer_size=100000):
    env = gym.make('Assault-v4', obs_type="ram")
    agent = NeuralNetwork().to(device)
    time_step = 0
    score = [0 for i in range(no_episodes)]
    for episode in range(no_episodes):
        observation, info = env.reset()
        current_state = agent.observation_to_state(observation).to(device).unsqueeze(1)
        previous_lives = 4
        while True:
            action = agent.predict(current_state / 255)
            previous_state = current_state
            previous_observation = observation
            current_observation, reward, done, _,info = env.step(action)
            current_state = agent.observation_to_state(current_observation).to(device).unsqueeze(1)
            time_step += 1
            score[episode] += reward
            reward /= 21
            if previous_lives > info["lives"]:
                previous_lives = info["lives"]
                reward = -1
            agent.update_replay_memory(previous_state, action, reward, current_state, done)
            agent.update_policy()
            if done:
                print("Episode no {} finished after {} timesteps".format(episode, time_step))
                print("Score: {}".format(score[episode]))
                break
        if save:
            torch.save(agent.state_dict(), "{}/custom_dqn_ep_{}.pth".format(MODELS_ROOT, episode))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(12, save=True)

# if __name__ == "__main__":
#     env = gym.make('Assault-v4', obs_type="ram")
#     net = NeuralNetwork()
#     print(net)
#     train(env, net)
#     torch.save(net, 'reinforcement.pth')
