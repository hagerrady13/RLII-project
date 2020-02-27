# actor_critic: section 13.5
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
import itertools
import plotting

matplotlib.style.use('ggplot')

seed = 99
rand_generator = np.random.RandomState(seed)

env = mc_env()

tile_coder = MountainCarTileCoder(iht_size=4096, num_tilings=8, num_tiles=8)

def convert_state(state):
    position, velocity = state
    tiles = tile_coder.get_tiles(position=position, velocity=velocity)
    return tiles

class Actor():
    def __init__(self, n_features):

        self.weights = np.zeros([env.action_count, n_features])
        self.step_size = 0.1/tile_coder.num_tilings

        self.actions = [0, 1, 2]

    def compute_softmax(self, active_tiles):
        state_action_preferences = self.weights[:, active_tiles].sum(axis=1)
        c = np.max(state_action_preferences)

        numerator = np.exp(state_action_preferences - c)
        denominator = np.sum(numerator)

        softmax_prob = numerator / denominator

        return softmax_prob

    def forward(self, x_s):
        active_tiles = convert_state(x_s)
        self.action_probs = self.compute_softmax(active_tiles)
        self.action = rand_generator.choice(env.action_count, p=self.action_probs)

        return self.action

    def backward(self, td_error, x_s):
        # tile coder derivative is 1 in case of active tiles, 0 otherwise
        current_tiles = convert_state(x_s)
        for a in self.actions:
            if a == self.action:
                self.weights[a][current_tiles] += self.step_size * td_error * (1 - self.action_probs[a])
            else:
                self.weights[a][current_tiles] += self.step_size * td_error * (0 - self.action_probs[a])

class Critic():
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.step_size = 1/tile_coder.num_tilings

    def forward(self, x_s):
        active_tiles = convert_state(x_s)
        return self.weights[active_tiles]

    def backward(self, td_error, x_s):
        current_tiles = convert_state(x_s)
        self.weights[current_tiles] += self.step_size * td_error


def agent():
    num_episodes = 500
    gamma = 1.0

    episodic_reward = np.zeros(num_episodes)

    actor = Actor(tile_coder.iht_size)
    critic = Critic(tile_coder.iht_size)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    env.env_init()

    for i_episode in range(num_episodes):

        state = env.env_start()

        for t in itertools.count():
            # Take a step
            action = actor.forward(state)

            reward, next_state, done = env.env_step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                value_next = critic.forward(state)*0
            else:
                value_next = critic.forward(next_state)

            td_target = reward + (gamma * value_next.sum())
            td_error = td_target - critic.forward(state).sum()

            critic.backward(td_error, state)
            actor.backward(td_error, state)

            episodic_reward[i_episode] += reward
            state = next_state

            if done:
                break
        # print(t, episodic_reward[i_episode])

    # plt.plot(np.arange(0, num_episodes), episodic_reward)
    print(episodic_reward)
    # plt.show()
    plotting.plot_episode_stats(stats, smoothing_window=10)

if __name__ == '__main__':
    agent()
