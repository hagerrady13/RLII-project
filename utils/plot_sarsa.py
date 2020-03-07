import matplotlib
import matplotlib.pyplot as plt
import numpy as np

alpha_hyper_parameters = [2**i for i in range(-8, 2)]
alpha_hyper_parameters.append(1.5)
# alpha_hyper_parameters.append(2)

means = np.load('../outputs/sarsa_param_study_means_0.0.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.0.npy')

means = np.insert(means, means.shape[0]-1, 227.6678)
stds = np.insert(stds, stds.shape[0]-1, 0.57)

fig, ax = plt.subplots(1)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)

x_axis = np.array([i for i in range(-8, 3)])

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink','gray', 'olive','cyan']
p2 = 0
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.0), color=colors[p2])
ax.text(x_axis[-2], means[-3], r'$\epsilon$=' + str(0.0), color=colors[p2], fontsize=11)

means = np.load('../outputs/sarsa_param_study_means_0.1.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.1.npy')

# means = np.insert(means, means.shape[0], 1000)
# stds = np.insert(stds, stds.shape[0], np.max(stds))
x_axis = np.array([i for i in range(-8, 2)])
p2 = 1
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.1), color=colors[p2])
ax.text(x_axis[-1], means[-1], r'$\epsilon$=' + str(0.1), color=colors[p2], fontsize=11)

means = np.load('../outputs/sarsa_param_study_means_0.05.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.05.npy')

# means = np.insert(means, means.shape[0], 1000)
# stds = np.insert(stds, stds.shape[0], np.max(stds))

p2 = 2
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.05), color=colors[p2])
ax.text(x_axis[-2], means[-1], r'$\epsilon$=' + str(0.05), color=colors[p2], fontsize=11)
# ax.set_xticklabels(x_axis)
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
# p2 = 1
# ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha$=' + str(x_axis[p2]), color=colors[p2])
# ax.text(x_axis[5], means[5, p2], r'$\alpha$=' + str(x_axis[p2]), color=colors[p2])

# ax.set_title("Paramater Study")

ax.set_title("Mountain Car - one-step Sarsa \n" + r'$\alpha$: Sarsa step size,'+ r'$\epsilon$: exploration paramater ', fontsize=10)
ax.set_ylabel("Steps\nper\nepisode\n\nAveraged\nover\nﬁrst 50\nepisodes\nand\n100\nruns", color='black', rotation=0, labelpad=15, fontsize=9)
ax.set_xlabel(r'$\alpha$' +" x number of tilings(8) - $\log_{2}$ scale", color='black')

# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.06), shadow=True, ncol=1)
# ax.legend(loc='best')
# ax.get_yaxis().set_ticks([])
ax.tick_params(axis='y', which='major', labelsize=5)
# ax.set_ylabel("Average steps over 50 episodes (100 runs)", color='black', fontsize=11)
# ax.set_xlabel("[Sarsa] "+ r'$\alpha$' +" x number of tilings", color='black', fontsize=11)
# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.06), shadow=True, ncol=1)
# ax.legend(loc='best')
ax.set_ylim(100, 1000)

fig.savefig("outputs/sarsa_alpha_eps.pdf", bbox_inches='tight')

plt.show()
