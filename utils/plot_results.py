import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.style.use('ggplot')

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
    ax.set_xlabel("Episode", fontsize=50)

    ax.legend(loc = 'best')
    ax.set_ylim(100, 600)

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

def plot_lr(arr1, arr2, arr3=None, num_episodes=500, n_runs=100):
    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    mean_steps = np.mean(arr1, axis=0)
    std_steps = np.std(arr1, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='blue')
    ax.text(430, 100, 'Sarsa, ' + r'$\epsilon$=0' , color='blue', fontsize=11)
    # ax.plot(x, mean_steps, lw=1, color='blue' , label='Sarsa ,'+ r'$\epsilon=0.0$')
    # ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    mean_steps = np.mean(arr3, axis=0)
    std_steps = np.std(arr3, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='orange')
    ax.text(300, 160, 'Sarsa, ' + r'$\epsilon$=0.1' , color='orange', fontsize=11)
    # ax.annotate('Sarsa,' + r'$\epsilon$=0.1', xy=(x[-3], mean_steps[-3]), xytext=(x[-3]+0.5, mean_steps[-3]+0.5),
    #             arrowprops=dict(facecolor='orange', shrink=0.05),)

    mean_steps = np.mean(arr2, axis=0)
    std_steps = np.std(arr2, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='red')
    ax.text(20, mean_steps[1], 'Actor-Critic', color='red', fontsize=11)

    ax.set_title("Mountain Car", fontsize=15)
    ax.set_ylabel("Steps\nper\nepisode\n\n(averaged\nover\n100 runs)", color='black', rotation=0, labelpad=30, fontsize=11)
    ax.set_xlabel("Episode", color='black', fontsize=11)

    # ax.legend(loc = 'best')
    ax.set_ylim(100, 500)

    plt.show()
    fig.savefig("../outputs/ac_vs_sarsa_lr_2.png", bbox_inches='tight')


def plot_td(arr1, arr2, arr3, num_episodes=500, n_runs=100):
    fig, ax = plt.subplots(1)
    x = np.arange(0, num_episodes)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    mean_steps = np.mean(arr1, axis=0)
    std_steps = np.std(arr1, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='blue')
    ax.text(400, 7, 'Sarsa,' + r'$\epsilon$=0' , color='blue', fontsize=11)

    mean_steps = np.mean(arr3, axis=0)
    std_steps = np.std(arr3, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='orange')
    ax.text(400, -40, 'Sarsa,' + r'$\epsilon$=0.1' , color='orange', fontsize=11)

    mean_steps = np.mean(arr2, axis=0)
    std_steps = np.std(arr2, axis=0)/np.sqrt(n_runs)

    ax.errorbar(x, mean_steps, std_steps, color='red')
    ax.text(x[160], mean_steps[160], 'Actor-Critic', color='red', fontsize=11)

    ax.set_title("Mountain Car", fontsize=15)
    ax.set_ylabel("TD\nError\n\n(averaged\nover\n100 runs)", color='black', rotation=0, labelpad=30, fontsize=11)
    ax.set_xlabel("Episode", color='black', fontsize=11)

    # ax.legend(loc = 'best')
    plt.axhline(y=0, color='black', linestyle='--', lw=0.5)
    ax.set_ylim(-500, 50)

    plt.show()
    fig.savefig("outputs/ac_vs_sarsa_td_error.pdf", bbox_inches='tight')

def plot_param_study(mean_steps, std_steps):
    fig, ax = plt.subplots(1)
    x = [0.0, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]#np.arange(0.1, 2.0, 0.1)
    # print(x[13])

    ax.plot(x, mean_steps, lw=2, color='blue' , label='1-step-SARSA')
    ax.fill_between(x, mean_steps - std_steps , mean_steps + std_steps, facecolor='blue', alpha=0.2)

    # ax.set_title("Average steps ")
    ax.set_ylabel("Average steps over 50 episodes")
    ax.set_xlabel("Epsilon")

    ax.legend(loc = 'best')
    ax.set_ylim(100, 1000)

    plt.show()

if __name__ == '__main__':
    arr1 = np.load('../outputs/SARSA_learning_rate.npy')
    arr2 = np.load('../outputs/AC_learning_rate.npy')
    arr3 = np.load('../outputs/SARSA_learning_rate_0.1.npy')

    # print(np.argmin(arr1))
    # print(arr2)
    plot_lr(arr1, arr2, arr3)

    # arr1 = np.load('outputs/SARSA_tderror.npy')
    # arr2 = np.load('outputs/AC_tderror.npy')
    # arr3 = np.load('outputs/SARSA_tderror_0.1.npy')
    # # # print(np.argmin(arr1))
    # # # print(arr2)
    # plot_td(arr1, arr2, arr3)
    # arr1 = np.load('SARSA_eps_means.npy')
    # arr2 = np.load('SARSA_eps_stds.npy')
    # print(np.argmin(arr1))
    # # print(arr2)
    # plot_param_study(arr1, arr2)
    # arr1 = np.load('sarsa_steps_alpha_0.1.npy')
    # arr2 = np.load('sarsa_steps_alpha_0.2.npy')
    # arr3 = np.load('sarsa_steps_eps0.npy')

    # plot_AC(arr1, arr2, arr3)

    # arr2 = np.load('ac_steps.npy')
    # arr1 = np.load('ac_steps_gamma.npy')
    # plot(arr1, arr2)

    # arr1 = np.load('sarsa_steps_eps0.npy')
    # arr2 = np.load('sarsa_steps_eps0.05.npy')
    # arr3 = np.load('sarsa_steps_eps0.1.npy')
    #
    # plot_SARSA(arr1, arr2, arr3)
