# Episodic one-step actor_critic: section 13.5
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from envs.mountain_car import Environment as mc_env
from envs.mc_tilecoder import MountainCarTileCoder
from utils import print_cost
import itertools
from mpl_toolkits.mplot3d.axes3d import Axes3D

matplotlib.use('Agg')

env = mc_env()


class ActorCritic:
    def __init__(self, seed, actor_step_size=0.5, critic_step_size=1.0):
        """
        Main class for actor-critic functions
        Args:
            seed: current seed for numpy
            actor_step_size:
            critic_step_size:
        """
        self.actions = [0, 1, 2]
        self.num_actions = len(self.actions)
        self.iht_size = 4096
        self.num_tilings = 8
        self.num_tiles = 8
        self.action_probs = None
        self.action = None

        self.actor_step_size = actor_step_size/self.num_tilings
        self.critic_step_size = critic_step_size/self.num_tilings

        self.rand_generator = np.random.RandomState(seed)
        np.random.seed(seed)

        self.tile_coder = MountainCarTileCoder(iht_size=self.iht_size , num_tilings=self.num_tilings, num_tiles=self.num_tiles)
        self.actor_weights = np.zeros([self.num_actions, self.iht_size])
        self.critic_weights = np.zeros(self.iht_size)

    def forward(self, x_s):
        """
        return action and value based on the current state, using softmax
        Args:
            x_s: state

        Returns:
            action, action_value
        """
        active_tiles = self.__convert_state(x_s)
        self.action_probs = self.__compute_softmax(active_tiles)
        self.action = self.rand_generator.choice(self.num_actions, p=self.action_probs)

        return self.action, self.critic_weights[active_tiles].sum()

    def backward(self, td_error, x_s):
        """
        performs backpropagation updates
        Args:
            td_error: different between estimate and target
            x_s: state
        Returns:
            None
        """
        # tile coder derivative is 1 in case of active tiles, 0 otherwise
        current_tiles = self.__convert_state(x_s)
        # update critic weights
        self.critic_weights[current_tiles] += self.critic_step_size * td_error

        # update actor weights
        for a in self.actions:
            if a == self.action:
                self.actor_weights[a][current_tiles] += self.actor_step_size * td_error * (1 - self.action_probs[a])
            else:
                self.actor_weights[a][current_tiles] += self.actor_step_size * td_error * (0 - self.action_probs[a])

    def get_critic(self, x_s):
        """
        retrieves value of the current state
        Args:
            x_s: state
        Returns:
            state value
        """
        active_tiles = self.__convert_state(x_s)
        return self.critic_weights[active_tiles].sum()

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

    def __compute_softmax(self, active_tiles):
        """
        private function for computing softmax probabilties
        Args:
            active_tiles: tiles of current state
        Returns: softmax probabilities
        """
        state_action_preferences = self.actor_weights[:, active_tiles].sum(axis=1)
        c = np.max(state_action_preferences)

        numerator = np.exp(state_action_preferences - c)
        denominator = np.sum(numerator)

        softmax_prob = numerator / denominator
        return softmax_prob

    def cost_to_go(self, pos, velo):
        """
        a utility function for computing the cost to go for plotting
        Args:
            pos: current position
            velo: current velocity
        Returns: value of the current state
        """
        tiles = self.tile_coder.get_tiles(position=pos, velocity=velo)
        return -(self.critic_weights[tiles].sum())

    def get_policy_information(self, pos, velo, index=0):
        """
        a utility function for computing the probability of each action for plotting
        Args:
            pos: current position
            velo: current velocity
        Returns: probability of a certain action given a state
        """
        tiles = self.tile_coder.get_tiles(position=pos, velocity=velo)
        action_probs = self.__compute_softmax(tiles)
        return action_probs[index]

def agent():
    """
    where all the action happens
    Returns:
    """
    num_episodes = 500
    gamma = 1.0
    n_runs = 1

    # for plotting
    plot_episodes = []#[1, 20, num_episodes - 1]
    fig = plt.figure(figsize=(20, 8))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    fig.suptitle('One-step Actor-Critic - Cost-to-go function on Mountain Car', fontsize=20, color='black')

    # for hyper-parameter search
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

        seed = 888 * run
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

                # compute target and update weights
                if done:
                    td_target = reward
                else:
                    next_value = actor_critic.get_critic(next_state)
                    td_target = reward + gamma * next_value

                td_error = td_target - current_value
                actor_critic.backward(td_error, state)

                episodic_reward[i_episode] += reward
                episodic_lengths[i_episode] += 1
                episodic_TDerror[i_episode] += td_error

                state = next_state

                if done:
                    break

            if i_episode in plot_episodes:
                print_cost(actor_critic, i_episode, axes[plot_episodes.index(i_episode)])

        all_rewards.append(episodic_reward)
        all_steps.append(episodic_lengths)
        all_td_errors.append(episodic_TDerror)

    # run_means.append(np.mean(all_steps))
    # run_stds.append(np.std(all_steps)/np.sqrt(n_runs))
    # alpha_actor.append(param_1)
    # alpha_critic.append(param_2)
    # plt.savefig('./outputs/sarsa_cost.png')
    # plt.close()

    np.save('outputs/AC_learning_rate.npy', np.array(all_steps))
    np.save('outputs/AC_tderror.npy', np.array(all_td_errors))
    np.save('outputs/AC_rewards.npy', np.array(all_rewards))


if __name__ == '__main__':
    agent()
