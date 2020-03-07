import matplotlib
import matplotlib.pyplot as plt
import numpy as np


actor_hyper_parameters = np.array([2**i for i in range(-8, 3)])
critic_hyper_parameters = np.array([2**i for i in range(-8, 2)])

means = np.zeros((actor_hyper_parameters.shape[0], critic_hyper_parameters.shape[0]))
stds = np.zeros((actor_hyper_parameters.shape[0], critic_hyper_parameters.shape[0]))

x = np.load('../outputs/alpha_actor_3.npy')
y = np.load('../outputs/alpha_critic_3.npy')
z = np.load('../outputs/ac_param_study_means_3.npy')
std = np.load('../outputs/ac_param_study_stds_3.npy')

means_extra = np.load('../outputs/ac_param_study_means_1.5.npy')
stds_extra = np.load('../outputs/ac_param_study_stds_1.5.npy')
print(means_extra)
print(stds_extra)
z = np.insert(z, z.shape[0] - critic_hyper_parameters.shape[0], means_extra)
std = np.insert(std, std.shape[0] - critic_hyper_parameters.shape[0], stds_extra)

for i in range(x.shape[0]):
    print(x[i], y[i], z[i], std[i])

for p1 in range(actor_hyper_parameters.shape[0]):
    for p2 in range(critic_hyper_parameters.shape[0]):
        means[p1, p2] = z[(p1*critic_hyper_parameters.shape[0]) + p2]
        stds[p1, p2] = std[(p1*critic_hyper_parameters.shape[0]) + p2]

fig, ax = plt.subplots(1)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

x_axis = np.array([i for i in range(-8, 3)])
# xticks(np.arange(0, 1, step=0.2))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'violet','gray', 'olive','turquoise']
p2 = 0
ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
ax.text(x_axis[5], means[5, p2], str(x_axis[p2]), color=colors[p2],fontsize=9)

p2 = 1
ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
ax.text(x_axis[5], means[5, p2], str(x_axis[p2]), color=colors[p2],fontsize=9)

p2 = critic_hyper_parameters.shape[0]-1
ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
ax.text(x_axis[0], means[0, p2], str(x_axis[p2]), color=colors[p2],fontsize=9)

p2 = critic_hyper_parameters.shape[0]-2
ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
ax.text(x_axis[-6], means[-6, p2], str(x_axis[p2]), color=colors[p2],fontsize=9)

for p2 in range(2, critic_hyper_parameters.shape[0]-2):
    ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label= r'$\alpha_{w}$='+str(x_axis[p2]), color=colors[p2])
    ax.text(x_axis[2], means[2, p2], str(x_axis[p2]), color=colors[p2], fontsize=9)

    print(p2, means[:, p2])

ax.text(x_axis[4], 900 , r'$\alpha_{w}$' +" x number of tilings(8)\n $\log_{2}$ scale", fontsize=9, color='blue')
# ax.annotate(r'$\alpha_{w}$' +" x number of \ntilings ($\log_{2}$ scale)", xy=(x_axis[5], 900), xytext=(x_axis[5], 900))
ax.xaxis.set_ticks(np.arange(-8, 3, 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
print(len(labels))
labels[0] = '$-8$'
labels[1] = '$-7$'
labels[2] = '$-6$'
labels[3] = '$-5$'
labels[4] = '$-4$'
labels[5] = '$-3$'
labels[6] = '$-2$'
labels[7] = '$-1$'
labels[8] = '$0$'
labels[9] = '$0.58$'
labels[10] = '$1$'
ax.set_xticklabels(labels)

ax.set_title("Mountain Car - one-step Actor-Critic \n " + r'$\alpha_{\theta}$: Actor step size, '+ r'$\alpha_{w}$: Critic step size ', fontsize=10)
# ax.set_ylabel("Steps\nper\nepisode\n\nAveraged\nover\nﬁrst 50\nepisodes\nand\n100\nruns", color='black', rotation=0, labelpad=15, fontsize=9)
ax.set_xlabel(r'$\alpha_{\theta}$' +" x number of tilings(8) - $\log_{2}$ scale", color='black')

# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.06), shadow=True, ncol=1)
# ax.legend(loc='best')
ax.get_yaxis().set_ticks([])
# ax.tick_params(axis='y', which='major', labelsize=5)
# ax.tick_params(axis='both', which='minor', labelsize=7)
ax.set_ylim(100, 1000)
fig.savefig("outputs/ac_alpha_2.pdf", bbox_inches='tight')
plt.show()
