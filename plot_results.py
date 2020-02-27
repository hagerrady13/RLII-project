import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

def plot_AC(arr1, arr2, arr3, n_runs=10, num_episodes=500):
    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    mean_steps = np.mean(arr1, axis=0)
    std_steps = np.std(arr1, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='blue' , label='AC-0.1')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    mean_steps = np.mean(arr2, axis=0)
    std_steps = np.std(arr2, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='red' , label='AC-0.2')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='red', alpha=0.2)

    mean_steps = np.mean(arr3, axis=0)
    std_steps = np.std(arr3, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='green' , label='AC-0.5')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='green', alpha=0.2)

    ax.set_title("Actor-Critic on Mountain Car")
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("Episode")

    ax.legend(loc = 'best')
    ax.set_ylim(100, 1000)

    plt.show()
    # fig.savefig("ac_vs_sarsa.pdf", bbox_inches='tight')

def plot_SARSA(arr1, arr2, arr3, n_runs=10, num_episodes=500):
    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    mean_steps = np.mean(arr1, axis=0)
    std_steps = np.std(arr1, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='blue' , label='eps-0')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    mean_steps = np.mean(arr2, axis=0)
    std_steps = np.std(arr2, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='red' , label='eps-0.05')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='red', alpha=0.2)

    mean_steps = np.mean(arr3, axis=0)
    std_steps = np.std(arr3, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='green' , label='eps-0.1')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='green', alpha=0.2)

    ax.set_title("SARSA on Mountain Car")
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("Episode")

    ax.legend(loc = 'best')
    ax.set_ylim(100, 1000)

    plt.show()

def plot(arr1, arr2, n_runs=10, num_episodes=500):
    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    mean_steps = np.mean(arr1, axis=0)
    std_steps = np.std(arr1, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='blue' , label='1-step-SARSA')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    mean_steps = np.mean(arr2, axis=0)
    std_steps = np.std(arr2, axis=0)/np.sqrt(n_runs)

    ax.plot(x, mean_steps, lw=2, color='red' , label='1-step-Actor-Critic')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='red', alpha=0.2)

    ax.set_title("SARSA vs. Actor-Critic on Mountain Car")
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("Episode")

    ax.legend(loc = 'best')
    # ax.set_ylim(100, 1000)

    plt.show()
    # fig.savefig("ac_vs_sarsa.pdf", bbox_inches='tight')

if __name__ == '__main__':
    arr2 = np.load('ac_TDerror.npy')
    arr1 = np.load('sarsa_TDerror.npy')
    plot(arr1, arr2)

    # arr1 = np.load('sarsa_steps_alpha_0.1.npy')
    # arr2 = np.load('sarsa_steps_alpha_0.2.npy')
    # arr3 = np.load('sarsa_steps_eps0.npy')
    #
    # plot_AC(arr1, arr2, arr3)

    # arr2 = np.load('ac_steps.npy')
    # arr1 = np.load('ac_steps_gamma.npy')
    # plot(arr1, arr2)

    # arr1 = np.load('sarsa_steps_eps0.npy')
    # arr2 = np.load('sarsa_steps_eps0.05.npy')
    # arr3 = np.load('sarsa_steps_eps0.1.npy')
    #
    # plot_SARSA(arr1, arr2, arr3)
