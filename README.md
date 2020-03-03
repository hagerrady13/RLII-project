# RL_2_project

Actor-Critic:
- Try different step_sizes, 100 runs: 0.9 actor, 1.8 critic

SARSA:
- Try different step_sizes, 100 runs: 1.4 is the best with no exploration
- Try different epsilon: the lower the exploration, the better. best eps=0.0

To compare speed:
- First 50 episodes over 100 runs: AC is better

To compare performance:
- 500 episodes of 100 runs: better at the beginning but same at end
- Compare TD error? they follow same dist. since they are both samples of the advantage function
- What about discounting/gamma? part of problem
- How to choose best alpha & epsilon at the same time?

Something like 12.5?

1)
Environment
Agent
Experiment
