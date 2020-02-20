# SARSA: section 6.1 / 10.1 (with function approximation)
from env import RandomWalkEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SARSA:
    def __init__(self, n_features):
        self.w = nn.Parameter(torch.randn(n_features, 2, requires_grad=True))
        self.optim = torch.optim.Adam(self.w, lr=0.1)
        self.loss = nn.MSELoss()

    def forward(self, x_s):
        # state_value = np.ones(env_size) # W_v * x
        return torch.mm(x_s, self.w)

    def backward(self, target, pred):
        value_loss = self.loss(td_target - pred)

        self.optim.zero_grad()
        value_loss.backward()
        self.optim.step()

def convert_state(state):
    x_s = np.zeros(env_size)
    x_s[state] = 1
    return x_s

def createEpsilonGreedyPolicy(q_value, eps=0.05, num_actions=2):

    def policyFunction(state):

        action_probs = np.ones(num_actions, dtype = float) * eps / num_actions

        best_action = np.argmax(q_value)

        action_probs[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        return action

    return policyFunction

def agent():
    env_size = 6
    num_episodes = 10
    time_steps = 10
    alpha_theta = 0.5
    alpha_w = 0.5
    gamma = 0.9

    env = RandomWalkEnv(size=env_size)
    episodic_reward = np.zeros(num_episodes)

    sarsa = SARSA(env_size)

    # x_s = # output of a function approximator or one-hot vector of states or (simply number of the state? )in the tabular case
    policy = createEpsilonGreedyPolicy(sarsa)

    for i_episode in range(num_episodes):

        state = env._reset()

        for t in range(time_steps):

            # Take a step
            q_value = sarsa.forward(convert_state(state))
            action = policy(q_value)
            next_state, reward, done, _ = env._step(action)

            if done:
                value_next = 0
            else:
                value_next = sarsa.forward(convert_state(next_state))

            td_target = reward + (gamma * value_next)
            value_current = sarsa.forward(convert_state(state))

            sarsa.backward(td_target, value_current)

            episodic_reward[i_episode] += reward
            state = next_state

            if done:
                break

if __name__ == '__name__':
    agent()
