import matplotlib.pyplot as plt
import numpy as np


def plot_3dgrid():
    actor_hyper_parameters = np.arange(0.1, 1.5, 0.1)
    critic_hyper_parameters = np.arange(0.1, 2.0, 0.1)

    x = np.load('alpha_actor.npy')
    y = np.load('alpha_critic.npy')
    z = np.load('ac_param_study_means.npy')
    std = np.load('ac_param_study_stds.npy')
    mat = np.zeros((len(actor_hyper_parameters), len(critic_hyper_parameters)))

    for p1 in range(actor_hyper_parameters.shape[0]):
        for p2 in range(critic_hyper_parameters.shape[0]):
            mat[p1, p2] = np.round(z[(p1*actor_hyper_parameters.shape[0]) + p2])

    indx = np.unravel_index(np.argmin(mat, axis=None), mat.shape)

    print("Actor Step size", actor_hyper_parameters[indx[0]])
    print("Critic Step size", critic_hyper_parameters[indx[1]])

    fig, ax = plt.subplots()

    ax.pcolormesh(np.arange(0.1, 2.1, 0.1), np.arange(0.1, 1.6, 0.1), mat, cmap='Greys', vmin=np.min(z), vmax=(np.max(z)))

    ax.set_ylabel('Actor step Size x number of tilings')
    ax.set_xlabel('Critic step Size x number of tilings')
    ax.set_title("Average of the first 50 episodes over 100 runs")

    for p1 in range(actor_hyper_parameters.shape[0]):
        for p2 in range(critic_hyper_parameters.shape[0]):
            ax.text(critic_hyper_parameters[p2], actor_hyper_parameters[p1], int(mat[p1, p2]))

    plt.show()


if __name__ == '__main__':
    plot_3dgrid()