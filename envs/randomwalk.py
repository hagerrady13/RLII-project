# example 9.1, in the book
import numpy as np

class RandomWalkEnv():
  def __init__(self, size = 6):
    self.action_space = [1, -1]
    self.size = size
    self.done = False
    self.action_space = 2
    self.observation_space = size
    print("Initialing Random walk Env with size: ", self.size)

  def step(self, action):
    reward = 0
    done = False

    if action == 0:
       self.state -= 1

    if action == 1:
        self.state += 1

    if self.state >= self.size:
        reward = 1
        self.done = True

    if self.state <= 0:
        self.done = True

    return np.array(self.state), reward, self.done, {}

  def reset(self):
    self.state = int(self.size / 2)#np.random.randint(1,self.size-1)
    print("starting at state: ", self.state, " with size: ", self.size)
    return self.state

  def _render(self):
    print("current state: ",self.state)
