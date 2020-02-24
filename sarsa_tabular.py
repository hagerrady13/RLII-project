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

Q = np.zeros((env.observation_space.n, env.action_space.n))

def epsilonGreedyPolicy(state, eps=0.1, num_actions=env.action_space.n):

    action_probs = np.ones(num_actions, dtype = float) * eps / num_actions
    best_action = np.argmax(Q[state], axis=0)
    action_probs[best_action] += (1.0 - eps)

    action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
    return action

def agent():
    # env_size = 6
    num_episodes = 300
    time_steps = 1000

    step_size = 0.5
    gamma = 1.0

    env = CliffWalkingEnv()
    episodic_reward = np.zeros(num_episodes)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # x_s = # output of a function approximator or one-hot vector of states or (simply number of the state? )in the tabular case
    env_size = env.observation_space.n

    for i_episode in range(num_episodes):

        state = env.reset()

        action = epsilonGreedyPolicy(state)

        for t in itertools.count():

            next_state, reward, done, _ = env.step(action)

            # Take a step
            next_action = epsilonGreedyPolicy(next_state)

            td_target = reward + (gamma * Q[next_state, next_action])

            Q[state, action] += step_size *(td_target - Q[state, action])

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            state = next_state
            action = next_action

            if done:
                break
        print(i_episode, episodic_reward[i_episode] )

    print(episodic_reward)
    plotting.plot_episode_stats(stats, smoothing_window=10)

if __name__ == '__main__':
    agent()
