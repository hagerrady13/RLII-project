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
    def __init__(self, step_size=1.0, eps=0.0):

        self.actions = [0, 1, 2]
        self.num_actions = len(self.actions)
        self.iht_size = 4096
        self.num_tilings = 8
        self.num_tiles = 8
        self.eps = eps
        self.step_size = step_size/self.num_tilings

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

    def epsilonGreedyPolicy(self, q_values):
        action_probs = np.ones(self.num_actions, dtype = float) * self.eps / self.num_actions
        best_action = self.__argmax(q_values)

        action_probs[best_action] += (1.0 - self.eps)
        action = np.random.choice(np.arange(self.num_actions), p = action_probs)

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
    num_episodes = 50
    gamma = 1.0
    n_runs = 100

    # alpha
    hyper_parameters_1 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.4, 0.5, 0.8]#np.arange(0.1, 2.0, 0.1) #EPS:[0.0, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.85]
    hyper_parameters_2 = [2**i for i in range(-10, 2)] #np.arange(0.1, 2.0, 0.1)

    run_means = []
    run_stds = []
    eps_values = []
    alpha_values = []

    for param_1 in hyper_parameters_1:
        for param_2 in hyper_parameters_2:
            all_rewards = []
            all_steps = []
            all_td_errors = []
            for run in range(n_runs):
                print("Run: ", run, param_1, param_2)
                episodic_reward = np.zeros(num_episodes)
                episodic_lengths = np.zeros(num_episodes)
                episodic_TDerror = np.zeros(num_episodes)

                seed = 999 * run
                np.random.seed(seed)

                sarsa_agent = SarsaAgent(step_size=param_2, eps=param_1)
                env.env_init()

                for i_episode in range(num_episodes):
                    state = env.env_start()

                    q_values = sarsa_agent.forward(state)
                    action, action_value = sarsa_agent.epsilonGreedyPolicy(q_values)

                    for t in range(1, 15000):
                        # print(i_episode, t)
                        # Take a step
                        reward, next_state, done = env.env_step(action)
                        if done:
                            td_error = reward - sarsa_agent.forward(state)[action]
                        else:
                            q_values_next = sarsa_agent.forward(next_state)
                            next_action, next_action_value = sarsa_agent.epsilonGreedyPolicy(q_values_next)
                            td_error = reward + (gamma * next_action_value) - sarsa_agent.forward(state)[action]

                        sarsa_agent.backward(td_error, state, action)

                        state = next_state
                        action = next_action
                        # action_value = next_action_value

                        episodic_reward[i_episode] += reward
                        episodic_lengths[i_episode] += 1
                        episodic_TDerror[i_episode] += td_error
                        # print(sarsa_agent.weights)
                        if done:
                            break

                # all_rewards.append(episodic_reward)
                # all_steps.append(episodic_lengths)
                all_steps.append(np.mean(episodic_lengths))
                # all_td_errors.append(episodic_TDerror)
        run_means.append(np.mean(all_steps))
        run_stds.append(np.std(all_steps)/np.sqrt(n_runs))
        eps_values.append(param_1)
        alpha_values.append(param_2)
    # fig, ax = plt.subplots(1)
    # x = hyper_parameters
    # # x = np.arange(0, num_episodes)
    #
    # # mean_steps = np.mean(all_steps, axis=0)
    # # std_steps = np.std(all_steps, axis=0)/np.sqrt(n_runs)
    # run_means = np.array(run_means)
    # # print(run_means)
    # run_stds = np.array(run_stds)
    # ax.plot(x, run_means, lw=1, color='red' , label='SARSA')
    # ax.fill_between(x, run_means - run_stds , run_means + run_stds, facecolor='red', alpha=0.2)
    #
    # # ax.set_title("1-Step-SARSA on Montain Car")
    # ax.set_ylabel("Steps per episode")
    # ax.set_xlabel("Alpha x number of tilings (8)")
    # # ax.legend(loc = 'best')
    # ax.set_ylim(100, 800)
    # fig.savefig("SARSA_alpha_param_study.pdf", bbox_inches='tight')
    # np.save('SARSA_eps_means.npy', np.array(run_means))
    # np.save('SARSA_eps_stds.npy', np.array(run_stds))
    np.save('alpha_sarsa.npy', np.array(alpha_values))
    np.save('eps_sarsa.npy', np.array(eps_values))
    np.save('sarsa_param_study_means.npy', np.array(run_means))
    np.save('sarsa_param_study_stds.npy', np.array(run_stds))

    # np.save('SARSA_learning_rate.npy', np.array(all_steps))

    # np.save('SARSA_tderror.npy', np.array(all_td_errors))

if __name__ == '__main__':
    agent()
