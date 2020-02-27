# SARSA: section 10.1 (with function approximation)
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

matplotlib.style.use('ggplot')

env = mc_env()

class SarsaAgent():
    def __init__(self):

        self.actions = [0, 1, 2]
        self.num_actions = len(self.actions)
        self.iht_size = 4096
        self.num_tilings = 8
        self.num_tiles = 8
        self.step_size = 0.5/self.num_tilings

        self.tile_coder = MountainCarTileCoder(iht_size=self.iht_size , num_tilings=self.num_tilings, num_tiles=self.num_tiles)
        self.weights = np.zeros([self.num_actions, self.tile_coder.iht_size])

    def forward(self, x_s):
        active_tiles = self.__convert_state(x_s)
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            action_values[a] = self.weights[a, active_tiles].sum()

        return action_values

    def backward(self, td_error, x_s, action):
        current_tiles = self.__convert_state(x_s)
        self.weights[action, current_tiles] += self.step_size * td_error

    def epsilonGreedyPolicy(self, q_values, eps=0.1):
        num_actions = env.action_count
        action_probs = np.ones(num_actions, dtype = float) * eps / num_actions
        best_action = self.__argmax(q_values)

        action_probs[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

        return action, q_values[action]

    def __convert_state(self, state):
        position, velocity = state
        tiles = self.tile_coder.get_tiles(position=position, velocity=velocity)
        return tiles

    # argmax that breaks ties
    def __argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

def agent():
    num_episodes = 500
    gamma = 1.0
    n_runs = 10

    all_rewards = []
    all_steps = []

    for run in range(n_runs):
        print("Run: ", run)
        episodic_reward = np.zeros(num_episodes)
        episodic_lengths = np.zeros(num_episodes)

        seed = 999 * run
        np.random.seed(seed)

        sarsa_agent = SarsaAgent()
        env.env_init()

        for i_episode in range(num_episodes):

            state = env.env_start()

            q_values = sarsa_agent.forward(state)
            action, action_value = sarsa_agent.epsilonGreedyPolicy(q_values)

            for t in itertools.count():

                # Take a step
                reward, next_state, done = env.env_step(action)
                if done:
                    td_target = reward
                else:
                    q_values_next = sarsa_agent.forward(next_state)
                    next_action, next_action_value = sarsa_agent.epsilonGreedyPolicy(q_values_next)
                    td_target = reward + (gamma * next_action_value) - action_value

                sarsa_agent.backward(td_target, state, action)

                state = next_state
                action = next_action
                action_value = next_action_value

                episodic_reward[i_episode] += reward
                episodic_lengths[i_episode] += 1

                if done:
                    break

        all_rewards.append(episodic_reward)
        all_steps.append(episodic_lengths)

    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    mean_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)/np.sqrt(n_runs)
    ax.plot(x, mean_steps, lw=1, color='blue' , label='SARSA')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    ax.set_title("SARSA on Montain Car")
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("Episode")
    ax.legend(loc = 'best')
    ax.set_ylim(100, 1000)

    plt.show()
    np.save('sarsa_steps.npy', np.array(all_steps))

if __name__ == '__main__':
    agent()
