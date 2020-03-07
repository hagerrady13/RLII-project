# actor_critic: section 13.5, not used in the project but was used to verify the tabular case of actor-critic
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from cliff_walking import CliffWalkingEnv
import itertools
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()


def convert_state(state, env_size=env.observation_space.n):
    x_s = np.zeros(env_size)
    x_s[state] = 1
    return x_s


class Actor(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.linear = nn.Linear(n_features, env.action_space.n, bias=False).double()
        self.linear.weight.data *=0

        self.softmax = nn.Softmax(dim=-1)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x_s):
        x = torch.from_numpy(convert_state(x_s)).double()
        # x = self.linear(x)
        # x = torch.relu(x)
        # Question, in the tabular case, shall I parameterize the policy and the value function too or not?
        action_probs = self.softmax(self.linear(x))

        return action_probs

    def backward(self, action_prob, td_error):
        # why is it -log(prob)? derive the equation
        policy_loss = -torch.log(action_prob) * td_error

        self.zero_grad()
        policy_loss.backward()
        self.optim.step()

class Critic(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.head = nn.Linear(n_features, 1, bias=False).double()
        # self.linear2 = nn.Linear(16, 2, bias=True).double()
        self.mseloss = nn.MSELoss()
        self.head.weight.data *=0

        self.optim = torch.optim.Adam(self.parameters(), lr=0.1)

    def forward(self, x_s):
        x = torch.from_numpy(convert_state(x_s)).double()

        return self.head(x)

    def backward(self, target, pred):
        value_loss = self.mseloss(target, pred)

        self.zero_grad()
        value_loss.backward(retain_graph=True)
        self.optim.step()


def agent():
    num_episodes = 500
    gamma = 1.0

    episodic_reward = np.zeros(num_episodes)

    env_size = env.observation_space.n

    actor = Actor(env_size)
    critic = Critic(env_size)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        state = env.reset()

        for t in itertools.count():

            # Take a step
            action_probs = actor.forward(state)
            action = np.random.choice(np.arange(len(action_probs.detach().numpy())), p=action_probs.detach().numpy())
            next_state, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                value_next = critic.forward(state)*0
            else:
                value_next = critic.forward(next_state)

            td_target = reward + (gamma * value_next)
            value_current = critic.forward(state)
            td_error = td_target - value_current

            critic.backward(td_target, value_current)

            actor.backward(torch.gather(input=action_probs, dim=0, index=torch.tensor(action)), td_error)
            episodic_reward[i_episode] += reward
            state = next_state

            if done:
                break
            print(t, done, episodic_reward[i_episode])

    print(episodic_reward)
    plotting.plot_episode_stats(stats, smoothing_window=10)


if __name__ == '__main__':
    agent()
