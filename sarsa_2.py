# SARSA: section 6.1 / 10.1 (with function approximation)
from env import RandomWalkEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gym
import matplotlib
import itertools

from cliff_walking import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

n_group = int(env.observation_space.n/8)
# 48*4
def convert_state(state, action=None, env_size=env.observation_space.n):
    x_s = np.zeros(n_group)
    x_s[int(state/8)] = 1.0
    return x_s

class SARSA(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.linear = nn.Linear(n_group, env.action_space.n, bias=False).double()
        self.optim = torch.optim.Adam(self.parameters(), 0.5)

        self.mseloss = nn.MSELoss()

        # self.linear.weight.data *= 0
        self.linear.weight.data.normal_(0.0, 1.0)

    def forward(self, s):
        x = torch.from_numpy(convert_state(s)).double()
        return self.linear(x)

    def backward(self, target, pred):
        self.optim.zero_grad()
        value_loss = self.mseloss(target, pred)
        value_loss.backward(retain_graph=True)
        self.optim.step()

# to break ties
def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)

def epsilonGreedyPolicy(q_values, eps=0.1, num_actions=env.action_space.n):
    # print(q_values)
    action_probs = np.ones(num_actions, dtype = float) * eps / num_actions
    best_action = argmax(q_values)
    # print(best_action, action_probs)
    action_probs[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
    # print(q_values)
    return action

def agent():
    # env_size = 6
    num_episodes = 300
    time_steps = 1000

    gamma = 1.0

    env = CliffWalkingEnv()
    episodic_reward = np.zeros(num_episodes)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # x_s = # output of a function approximator or one-hot vector of states or (simply number of the state? )in the tabular case
    env_size = env.observation_space.n
    # print(env_size)
    # print(env.action_space.n)
    # exit(0)

    sarsa = SARSA(env_size)

    for i_episode in range(num_episodes):

        state = env.reset()

        q_values = sarsa.forward(state)
        action = epsilonGreedyPolicy(q_values.detach().numpy())

        for t in itertools.count():

            next_state, reward, done, _ = env.step(action)
            # Take a step
            if done:
                q_values_next = sarsa.forward(state)*0
                td_target = reward + (gamma * q_values_next[action])
                sarsa.backward(td_target, q_values[action])
            else:
                q_values_next = sarsa.forward(next_state)
                next_action = epsilonGreedyPolicy(q_values_next.detach().numpy())

                td_target = reward + (gamma * q_values_next[next_action])

                sarsa.backward(td_target, q_values[action])

                state = next_state
                action = next_action
                q_values = q_values_next

            # print(sarsa.linear.weight.data)
            # print(q_values)
            episodic_reward[i_episode] += reward
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
        print(i_episode, episodic_reward[i_episode] )

    # plt.plot(np.arange(0, num_episodes), episodic_reward)
    print(episodic_reward)
    # plt.show()
    # plotting.plot_episode_stats(stats, smoothing_window=10)

if __name__ == '__main__':
    agent()
