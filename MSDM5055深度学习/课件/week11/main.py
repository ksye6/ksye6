import gymnasium as gym
from gymnasium.utils.save_video import save_video
import numpy as np
import random
import torch
import torch.optim as optim

from Qlearning import Qnetwork, ReplayBuff


def train(env, replayBuff, Qnet, targetNet, high, low, batchSize=64, learningRate=5e-3, gamma=0.99, Nepisode=4000, Nsteps=3000, Nexplore=5e3, eps=[0.01, 0.0001], render=False, savePerEp=100):

    optimizer = optim.AdamW(Qnet.parameters(), lr=learningRate, amsgrad=True)
    epsilon = eps[0]
    triggerFn = lambda i: i % (savePerEp / 20) == 0
    for n in range(Nepisode):
        s, info = env.reset()
        for j in range(Nsteps):

            with torch.no_grad():
                Qvalue = Qnet.pred(torch.from_numpy(s.reshape(1, Qnet.stateDim)))
                Qmax = Qvalue.max(1)[0]
                aIndex = Qvalue.max(1)[1]
            if random.random() <= epsilon:
                aIndex = torch.tensor([[env.action_space.sample()]], dtype=torch.long)

            # update epsilon
            if epsilon > eps[1]:
                epsilon -= (eps[0] - eps[1]) / Nexplore

            s2, r, term, trun, info = env.step(aIndex.item())

            replayBuff.add(s, aIndex, r, term, s2)

            if replayBuff.size > batchSize:
                sbatch, abatch, rbatch, termbatch, s2batch = replayBuff.sample(batchSize)

                sbatch = torch.tensor(sbatch)
                abatch = torch.tensor(abatch).unsqueeze(-1)
                rbatch = torch.tensor(rbatch).unsqueeze(-1)
                termbatch = torch.tensor(termbatch)
                s2batch = torch.tensor(s2batch)

                targetQ = torch.zeros(batchSize)
                with torch.no_grad():
                    nonFinalTargetQ = targetNet.pred(s2batch[~termbatch]).max(1)[0]
                targetQ[~termbatch] = nonFinalTargetQ
                targetQ = targetQ.unsqueeze(-1)

                predQ = Qnet.pred(sbatch).gather(1, abatch)

                expect = (targetQ * gamma) + rbatch

                loss = ((expect - predQ)**2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                targetNetState = targetNet.state_dict()
                QnetState = Qnet.state_dict()

                for key in QnetState:
                    targetNetState[key] = QnetState[key]*Qnet.tau + targetNetState[key]*(1-Qnet.tau)
                targetNet.load_state_dict(targetNetState)

            s = s2
            print("episode", n, ", timeStep", j, ", state", s, ", action", aIndex.item(), ", reward", r, ", Qmax ", Qmax.item())


            if term:
                if render:
                    save_video(
                        env.render(),
                        "videos",
                        episode_trigger=triggerFn,
                        fps=env.metadata["render_fps"],
                        step_starting_index=0,
                        episode_index=n
                    )
                print("Breaking one episode")
                break

        if n % savePerEp == 0:
            torch.save(Qnet, "Qnetwork.pth")
    env.close()


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array_list")

    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n

    high = env.observation_space.high
    low = env.observation_space.low

    qnet = Qnetwork(stateDim, actionDim, tau=0.01)
    targetQnet = Qnetwork(stateDim, actionDim, tau=0.1)
    targetQnet.load_state_dict(qnet.state_dict())
    replayBuff = ReplayBuff(10000)

    train(env, replayBuff, qnet, targetQnet, high, low, render=True)

