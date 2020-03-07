import matplotlib
import matplotlib.pyplot as plt
import numpy as np

alpha_hyper_parameters = np.arange(0.3, 1.5, 0.1)
# alpha_hyper_parameters.append(2)

means = np.load('../outputs/sarsa_param_study_means_0.0_.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.0_.npy')

fig, ax = plt.subplots(1)

x_axis = np.arange(0.3, 1.5, 0.1)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink','gray', 'olive','cyan']
p2 = 0
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.0), color=colors[p2])
ax.text(x_axis[-2], means[-3], r'$\epsilon$=' + str(0.0), color=colors[p2], fontsize=11)

means = np.load('../outputs/sarsa_param_study_means_0.1_.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.1_.npy')

# means = np.insert(means, means.shape[0], 1000)
# stds = np.insert(stds, stds.shape[0], np.max(stds))
# x_axis = np.array([i for i in range(-8, 2)])
p2 = 1
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.1), color=colors[p2])
ax.text(x_axis[-4], means[-4], r'$\epsilon$=' + str(0.1), color=colors[p2], fontsize=11)

means = np.load('../outputs/sarsa_param_study_means_0.05_.npy')
stds = np.load('../outputs/sarsa_param_study_stds_0.05_.npy')

# means = np.insert(means, means.shape[0], 1000)
# stds = np.insert(stds, stds.shape[0], np.max(stds))

p2 = 2
ax.errorbar(x_axis, means, yerr=stds, label=r'$\epsilon$=' + str(0.05), color=colors[p2])
ax.text(x_axis[-2], means[-1], r'$\epsilon$=' + str(0.05), color=colors[p2], fontsize=11)
# ax.set_xticklabels(x_axis)
ax.xaxis.set_ticks(np.arange(0.3, 1.5, 0.1))
labels = [item.get_text() for item in ax.get_xticklabels()]
print(len(labels))
labels[0] = '$0.3$'
labels[1] = '$0.4$'
labels[2] = '$0.5$'
labels[3] = '$0.6$'
labels[4] = '$0.7$'
labels[5] = '$0.8$'
labels[6] = '$0.9$'
labels[7] = '$1.0$'
labels[8] = '$1.1$'
labels[9] = '$1.2$'
labels[10] = '$1.3$'
labels[11] = '$1.4$'
ax.set_xticklabels(labels)
# p2 = 1
# ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha$=' + str(x_axis[p2]), color=colors[p2])
# ax.text(x_axis[5], means[5, p2], r'$\alpha$=' + str(x_axis[p2]), color=colors[p2])

# ax.set_title("Paramater Study")
ax.set_ylabel("Average steps over 50 episodes (50 runs)", color='black', fontsize=11)
ax.set_xlabel("[Sarsa] "+ r'$\alpha$' +" x number of tilings", color='black', fontsize=11)
# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.06), shadow=True, ncol=1)
# ax.legend(loc='best')
ax.set_ylim(100, 1000)

fig.savefig("outputs/sarsa_alpha_eps_2.pdf", bbox_inches='tight')

plt.show()
