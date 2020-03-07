# semi-gradient SARSA: section 10.1 (with function approximation)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import print_cost
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
import itertools

matplotlib.use('Agg')

env = mc_env()


class SarsaAgent:
    def __init__(self, step_size=1.0, eps=0.0):
        """
        Main class for Sarsa functions
        Args:
            step_size: 
            eps: 
        """
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
        """
        return action values based on the current state
        Args:
            x_s: state

        Returns:
            action_values
        """
        active_tiles = self.__convert_state(x_s)
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            action_values[a] = self.weights[a, active_tiles].sum()

        return action_values

    def backward(self, td_error, x_s, action):
        """
        performs backpropagation updates
        Args:
            td_error:
            x_s: state
            action:

        Returns:
            None
        """
        current_tiles = self.__convert_state(x_s)
        self.weights[action, current_tiles] += self.step_size * td_error

    def epsilon_greedy_policy(self, q_values):
        """
        given the action values, it uses epsilon greedy policy to find the action
        Args:
            q_values: action values
        Returns:
            action, action value
        """
        action_probs = np.ones(self.num_actions, dtype = float) * self.eps / self.num_actions
        best_action = self.__argmax(q_values)

        action_probs[best_action] += (1.0 - self.eps)
        action = np.random.choice(np.arange(self.num_actions), p = action_probs)

        return action, q_values[action]

    def __convert_state(self, state):
        """
        retrieves the active tiles of a state
        Args:
            state: current state
        Returns:
            tiles
        """
        position, velocity = state
        tiles = self.tile_coder.get_tiles(position=position, velocity=velocity)
        return tiles

    def __argmax(self, q_values):
        """
        argmax that breaks ties
        Args:
            q_values:
        Returns:

        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

    def cost_to_go(self, pos, velo):
        """
        a utility function for computing the cost to go for plotting
        Args:
            pos: current position
            velo: current velocity
        Returns: value of the current state
        """
        tiles = self.tile_coder.get_tiles(position=pos, velocity=velo)
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            action_values[a] = self.weights[a, tiles].sum()

        return -np.max(action_values)

    def get_policy_information(self, pos, velo, index=0):
        """
        a utility function for computing the probability of each action for plotting
        Args:
            pos: current position
            velo: current velocity
        Returns: probability of a certain action given a state
        """
        tiles = self.tile_coder.get_tiles(position=pos, velocity=velo)
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            action_values[a] = self.weights[a, tiles].sum()

        action_probs = np.ones(self.num_actions, dtype = float) * self.eps / self.num_actions
        best_action = self.__argmax(action_values)

        action_probs[best_action] += (1.0 - self.eps)
        return action_probs[index]


def agent():
    """
    where all the action happens
    Returns:

    """
    num_episodes = 500
    gamma = 1.0
    n_runs = 1

    plot_episodes = [] #[1, 20, num_episodes - 1]
    fig = plt.figure(figsize=(20, 8))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    fig.suptitle('One-step Sarsa - Cost-to-go function on Mountain Car', fontsize=20, color='black')

    # for hyper-parameter search
    hyper_parameters_1 = [0.0, 0.05, 0.1]
    hyper_parameters_2 = [2**i for i in range(-8, 2)]

    run_means = []
    run_stds = []
    eps_values = []
    alpha_values = []

    # for param_1 in hyper_parameters_1:
    # for param_2 in hyper_parameters_2:
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

        sarsa_agent = SarsaAgent(step_size=0.5, eps=0.0)
        env.env_init()

        for i_episode in range(num_episodes):
            if i_episode in plot_episodes:
                print_cost(sarsa_agent, i_episode, axes[plot_episodes.index(i_episode)])

            state = env.env_start()

            q_values = sarsa_agent.forward(state)
            action, action_value = sarsa_agent.epsilon_greedy_policy(q_values)

            for t in itertools.count():
                # Take a step
                reward, next_state, done = env.env_step(action)
                if done:
                    td_error = reward - sarsa_agent.forward(state)[action]
                else:
                    q_values_next = sarsa_agent.forward(next_state)
                    next_action, next_action_value = sarsa_agent.epsilon_greedy_policy(q_values_next)
                    td_error = reward + (gamma * next_action_value) - sarsa_agent.forward(state)[action]

                sarsa_agent.backward(td_error, state, action)

                state = next_state
                action = next_action

                episodic_reward[i_episode] += reward
                episodic_lengths[i_episode] += 1
                episodic_TDerror[i_episode] += td_error

                if done:
                    break

        all_rewards.append(episodic_reward)
        all_steps.append(episodic_lengths)
        # all_steps.append(np.mean(episodic_lengths))
        all_td_errors.append(episodic_TDerror)

    # run_means.append(np.mean(all_steps))
    # run_stds.append(np.std(all_steps)/np.sqrt(n_runs))
    # eps_values.append(param_1)
    # alpha_values.append(param_2)

    # plt.savefig('./outputs/sarsa_cost.png')
    # plt.close()

    np.save('outputs/SARSA_learning_rate_0.1.npy', np.array(all_steps))
    np.save('outputs/SARSA_tderror_0.1.npy', np.array(all_td_errors))
    np.save('outputs/SARSA_rewards_0.1.npy', np.array(all_rewards))

if __name__ == '__main__':
    agent()
