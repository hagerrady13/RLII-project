# semi-gradient SARSA: section 10.1 (with function approximation)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import print_cost
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
import itertools

#matplotlib.use('Agg')

env = mc_env()


class qlearningAgnet:
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
        best_action = self.argmax(q_values)

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

    def argmax(self, q_values):
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
        best_action = self.argmax(action_values)

        action_probs[best_action] += (1.0 - self.eps)
        return action_probs[index]


class Trajectory(object):
    def __init__(self, state, action, reward, next_state, next_action):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action


def make_traj(state, action, reward, next_state, next_action):
    traj = Trajectory(state, action, reward, next_state, next_action)
    return traj


def agent():
    """
    where all the action happens
    Returns:

    """
    num_episodes = 100
    gamma = 1.0
    n_runs = 1

    plot_episodes = [] #[1, 20, num_episodes - 1]
    # fig = plt.figure(figsize=(20, 8))
    # axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    # fig.suptitle('One-step Sarsa - Cost-to-go function on Mountain Car', fontsize=20, color='black')

    all_steps = []
    all_td_errors = []
    episodic_reward = np.zeros((n_runs, num_episodes))
    # all_history = np.empty((n_runs, num_episodes), dtype=np.object)

    for run in range(n_runs):
        print("Run: ", run)
        episodic_lengths = np.zeros(num_episodes)
        episodic_TDerror = np.zeros(num_episodes)

        seed = 888 * run
        np.random.seed(seed)

        agent = qlearningAgnet(step_size=0.5, eps=0.05)
        env.env_init()

        for i_episode in range(num_episodes):
            print(i_episode)
            # if i_episode in plot_episodes:
            #     print_cost(agent, i_episode, axes[plot_episodes.index(i_episode)])

            # all_history[run, i_episode] = []

            state = env.env_start()

            q_values = agent.forward(state)

            action, action_value = agent.epsilon_greedy_policy(q_values)

            for t in range(1, 5000):
                # Take a step
                reward, next_state, done = env.env_step(action)
                if done:
                    td_error = reward - agent.forward(state)[action]
                    # all_history[run, i_episode].append(make_traj(state, action, -1, [], -1))
                else:
                    q_values_next = agent.forward(next_state)
                    # next_action, next_action_value = agent.epsilon_greedy_policy(q_values_next)
                    next_action = agent.argmax(q_values_next)
                    next_action_value = q_values_next[next_action]

                    td_error = reward + (gamma * next_action_value) - agent.forward(state)[action]
                    # all_history[run, i_episode].append(make_traj(state, action, reward, next_state, next_action))

                agent.backward(td_error, state, action)

                state = next_state
                action , _ = agent.epsilon_greedy_policy(q_values_next)


                episodic_reward[run, i_episode] += reward
                episodic_lengths[i_episode] += 1
                episodic_TDerror[i_episode] += td_error

                if done:
                    break

        all_steps.append(episodic_lengths)
        # all_steps.append(np.mean(episodic_lengths))
        all_td_errors.append(episodic_TDerror)


    import matplotlib.pyplot as plt
    episodic_reward = np.mean(episodic_reward, axis=0)
    plt.plot(np.arange(0, num_episodes), episodic_reward)
    plt.ylim(-500, 0)
    plt.show()


if __name__ == '__main__':
    agent()
