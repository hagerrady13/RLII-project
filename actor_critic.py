# actor_critic: section 13.5
from env import RandomWalkEnv
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor:
    def __init__(self, n_features):
        self.w = nn.Parameter(torch.randn(n_features, 2), requires_grad=True))
        self.optim = torch.optim.Adam(self.w, lr=0.3)

    def forward(self, x_s):
        policy_function = torch.mm(x_s, self.w) # x = (N), w_policy = (N)
        # Question, in the tabular case, shall I parameterize the policy and the value function too or not?
        action_probs = F.softmax(policy_function)

        return action_probs

    def backward(self, action_prob, td_error):
        # why is it -log(prob)? derive the equation
        policy_loss = -torch.log(action_prob) * td_error

        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()

class Critic:
    def __init__(self, n_features):
        self.w = nn.Parameter(torch.randn(n_features, requires_grad=True))
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

def agent():
    env_size = 6
    num_episodes = 10
    time_steps = 10
    alpha_theta = 0.5
    alpha_w = 0.5
    gamma = 0.9

    env = RandomWalkEnv(size=env_size)
    episodic_reward = np.zeros(num_episodes)

    actor = Actor(env_size)
    critic = Critic(env_size)

    # x_s = # output of a function approximator or one-hot vector of states or (simply number of the state? )in the tabular case

    for i_episode in range(num_episodes):

        state = env._reset()

        for t in range(time_steps):

            # Take a step
            action_probs = actor.forward(convert_state(state))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env._step(action)

            if done:
                value_next = 0
            else:
                value_next = critic.forward(convert_state(next_state))

            td_target = reward + (gamma * value_next)
            value_current = critic.predict(convert_state(state))
            td_error = td_target - value_current

            actor.backward(action_probs[action], td_error)
            critic.backward(td_target, value_current)

            episodic_reward[i_episode] += reward
            state = next_state

            if done:
                break


if __name__ == '__name__':
    agent()
