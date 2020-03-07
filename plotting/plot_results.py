# plots learning rate and TD errors
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    arr1 = np.load('../outputs/SARSA_learning_rate.npy')
    arr2 = np.load('../outputs/AC_learning_rate.npy')
    arr3 = np.load('../outputs/SARSA_learning_rate_0.1.npy')

    plot_lr(arr1, arr2, arr3)

    # arr1 = np.load('../outputs/SARSA_tderror.npy')
    # arr2 = np.load('../outputs/AC_tderror.npy')
    # arr3 = np.load('../outputs/SARSA_tderror_0.1.npy')

    # plot_td(arr1, arr2, arr3)
