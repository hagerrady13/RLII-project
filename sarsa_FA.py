# SARSA: section 10.1 (with function approximation)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from cliff_walking import CliffWalkingEnv
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
import itertools
import plotting

matplotlib.style.use('ggplot')

env = mc_env()

tile_coder = MountainCarTileCoder(iht_size=4096, num_tilings=8, num_tiles=8)

seed = 99
np.random.seed(seed)

def convert_state(state):
    position, velocity = state
    tiles = tile_coder.get_tiles(position=position, velocity=velocity)
    return tiles

class SARSA():
    def __init__(self, n_features):

        self.weights = np.zeros([env.action_count, n_features])
        self.step_size = 0.5/tile_coder.num_tilings
        self.actions = [0, 1, 2]
        self.num_actions = len(self.actions)

    def forward(self, x_s):
        active_tiles = convert_state(x_s)
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            action_values[a] = self.weights[a, active_tiles].sum()

        return action_values

    def lookup(self, x_s, a):
        active_tiles = convert_state(x_s)
        return self.weights[a, active_tiles].sum()

    def backward(self, td_error, x_s, action):
        current_tiles = convert_state(x_s)
        self.weights[action, current_tiles] += self.step_size * td_error

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

def epsilonGreedyPolicy(q_values, eps=0.0):
    num_actions = env.action_count
    action_probs = np.ones(num_actions, dtype = float) * eps / num_actions
    best_action = argmax(q_values)

    action_probs[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

    return action, q_values[action]

def agent():
    num_episodes = 50
    time_steps = 1000

    gamma = 1.0

    episodic_reward = np.zeros(num_episodes)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    env_size = tile_coder.iht_size

    sarsa = SARSA(env_size)

    env.env_init()
    for i_episode in range(num_episodes):

        state = env.env_start()

        q_values = sarsa.forward(state)
        action, action_value = epsilonGreedyPolicy(q_values)

        for t in itertools.count():

            reward, next_state, done = env.env_step(action)
            # Take a step
            if done:
                td_target = reward
            else:
                q_values_next = sarsa.forward(next_state)
                next_action, next_action_value = epsilonGreedyPolicy(q_values_next)
                td_target = reward + (gamma * next_action_value) - action_value

            sarsa.backward(td_target, state, action)

            state = next_state
            action = next_action
            action_value = next_action_value

            episodic_reward[i_episode] += reward
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
        print(t, episodic_reward[i_episode] )

    # plt.plot(np.arange(0, num_episodes), episodic_reward)
    print(episodic_reward)
    # plt.show()
    plotting.plot_episode_stats(stats, smoothing_window=10)

if __name__ == '__main__':
    agent()
