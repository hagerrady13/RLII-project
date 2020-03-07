# actor_critic: section 13.5
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
import itertools

matplotlib.style.use('ggplot')

env = mc_env()

class ActorCritic():
    def __init__(self, seed, actor_step_size=0.5, critic_step_size=1.0):

        self.actions = [0, 1, 2]
        self.num_actions = len(self.actions)
        self.iht_size = 4096
        self.num_tilings = 8
        self.num_tiles = 8

        self.actor_step_size = actor_step_size/self.num_tilings
        self.critic_step_size = critic_step_size/self.num_tilings

        self.rand_generator = np.random.RandomState(seed)
        np.random.seed(seed)

        self.tile_coder = MountainCarTileCoder(iht_size=self.iht_size , num_tilings=self.num_tilings, num_tiles=self.num_tiles)
        self.actor_weights = np.zeros([self.num_actions, self.iht_size])
        self.critic_weights = np.zeros(self.iht_size)

    def forward(self, x_s):
        active_tiles = self.__convert_state(x_s)
        self.action_probs = self.__compute_softmax(active_tiles)
        self.action = self.rand_generator.choice(self.num_actions, p=self.action_probs)

        return self.action, self.critic_weights[active_tiles].sum()

    def backward(self, td_error, x_s):
        # tile coder derivative is 1 in case of active tiles, 0 otherwise
        current_tiles = self.__convert_state(x_s)
        # update critic
        self.critic_weights[current_tiles] += self.critic_step_size * td_error

        # update actor
        for a in self.actions:
            if a == self.action:
                self.actor_weights[a][current_tiles] += self.actor_step_size * td_error * (1 - self.action_probs[a])
            else:
                self.actor_weights[a][current_tiles] += self.actor_step_size * td_error * (0 - self.action_probs[a])

    def get_critic(self, x_s):
        active_tiles = self.__convert_state(x_s)
        return self.critic_weights[active_tiles].sum()

    def __compute_softmax(self, active_tiles):
        state_action_preferences = self.actor_weights[:, active_tiles].sum(axis=1)
        c = np.max(state_action_preferences)

        numerator = np.exp(state_action_preferences - c)
        denominator = np.sum(numerator)

        softmax_prob = numerator / denominator

        return softmax_prob

    def __convert_state(self, state):
        position, velocity = state
        tiles = self.tile_coder.get_tiles(position=position, velocity=velocity)
        return tiles

def agent():
    num_episodes = 500
    gamma = 1.0
    n_runs = 5

    # alpha
    actor_hyper_parameters = np.arange(0.1, 1.5, 0.1)
    critic_hyper_parameters = np.arange(0.1, 2.0, 0.1)
    run_means = []
    run_stds = []
    alpha_actor = []
    alpha_critic = []

    # for param_1 in actor_hyper_parameters:
        # for param_2 in critic_hyper_parameters:
    all_rewards = []
    all_steps = []
    all_td_errors = []
    for run in range(n_runs):
        print("Run: ", run)
        episodic_reward = np.zeros(num_episodes)
        episodic_lengths = np.zeros(num_episodes)
        episodic_TDerror = np.zeros(num_episodes)

        seed = 999 * run
        np.random.seed(seed)

        actor_critic = ActorCritic(seed=seed)
        env.env_init()

        for i_episode in range(num_episodes):

            state = env.env_start()

            for t in range(1, 5000):
                # print(t, i_episode)
                # Take a step
                action, current_value = actor_critic.forward(state)

                reward, next_state, done = env.env_step(action)

                if done:
                    td_target = reward
                else:
                    next_value = actor_critic.get_critic(next_state)
                    td_target = reward + gamma * next_value

                td_error = td_target - current_value

                episodic_reward[i_episode] += reward
                episodic_lengths[i_episode] += 1
                episodic_TDerror[i_episode] += td_error

                actor_critic.backward(td_error, state)

                episodic_reward[i_episode] += reward

                state = next_state

                if done:
                    break

        all_rewards.append(episodic_reward)
        all_steps.append(episodic_lengths)
        all_td_errors.append(episodic_TDerror)

            # run_means.append(np.mean(all_steps))
            # run_stds.append(np.std(all_steps)/np.sqrt(n_runs))
            # alpha_actor.append(param_1)
            # alpha_critic.append(param_2)
        run_means.append(actor_critic.actor_weights)
        run_stds.append(actor_critic.critic_weights)

    np.save('outputs/actor_weights.npy', np.mean(np.array(run_means)))
    np.save('outputs/critic_weights.npy', np.mean(np.array(run_stds)))
    exit(0)

    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    mean_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)/np.sqrt(n_runs)

    print(mean_steps)
    print(std_steps)

    ax.plot(x, mean_steps, lw=1, color='red' , label='SARSA')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='red', alpha=0.2)
    ax.set_title("1-Step-SARSA on Montain Car")
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("Alpha x number of tilings (8)")
    ax.legend(loc = 'best')
    ax.set_ylim(100, 800)
    plt.show()

    np.save('outputs/AC_learning_rate.npy', np.array(all_steps))
    np.save('outputs/AC_tderror.npy', np.array(all_td_errors))
    np.save('outputs/AC_rewards.npy', np.array(all_rewards))

if __name__ == '__main__':
    agent()
