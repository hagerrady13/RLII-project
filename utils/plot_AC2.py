import matplotlib
import matplotlib.pyplot as plt
import numpy as np

actor_hyper_parameters = np.arange(0.5, 1.9, 0.1)
critic_hyper_parameters = np.arange(0.3, 1.5, 0.1)

means = np.zeros((actor_hyper_parameters.shape[0] , critic_hyper_parameters.shape[0]))
stds = np.zeros((actor_hyper_parameters.shape[0], critic_hyper_parameters.shape[0]))

x = np.load('../outputs/alpha_actor_4.npy')
y = np.load('../outputs/alpha_critic_4.npy')
z = np.load('../outputs/ac_param_study_means_4.npy')
std = np.load('../outputs/ac_param_study_stds_4.npy')

for i in range(x.shape[0]):
    print(x[i], y[i], z[i], std[i])

for p1 in range(actor_hyper_parameters.shape[0] ):
    for p2 in range(critic_hyper_parameters.shape[0]):
        means[p1, p2] = z[(p1*critic_hyper_parameters.shape[0]) + p2]
        stds[p1, p2] = std[(p1*critic_hyper_parameters.shape[0]) + p2]

fig, ax = plt.subplots(1)

x_axis = np.arange(0.5, 1.9, step=0.1)
# xticks(np.arange(0, 1, step=0.2))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink','gray', 'olive','cyan', 'black', 'yellow']
# p2 = 0
# ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
# ax.text(x_axis[5], means[5, p2], r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
#
# p2 = 1
# ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
# ax.text(x_axis[5], means[5, p2], r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
# p2 = critic_hyper_parameters.shape[0]-1
# ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label=r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])
# ax.text(x_axis[0], means[0, p2], r'$\alpha_{w}$=' + str(x_axis[p2]), color=colors[p2])

for p2 in range(critic_hyper_parameters.shape[0]):
    print(p2, means[:, p2])

    # ax.errorbar(x_axis, means[:, p2], yerr=stds[:, p2], label= r'$\alpha_{w}$='+str(x_axis[p2]), color=colors[p2])
    # ax.text(x_axis[2], means[2, p2], r'$\alpha_{w}$='+str(x_axis[p2]), color=colors[p2])
    ax.plot(x_axis, means[:, p2], lw=1, label= r'$\alpha$'+str(x_axis[p2]))
    # ax.fill_between(x_axis, means[:, p2] - stds[:, p2] , means[:, p2] + stds[:, p2], alpha=0.2)

ax.xaxis.set_ticks(np.arange(0.5, 2.0, 0.1))
labels = [item.get_text() for item in ax.get_xticklabels()]
# print(len(labels))
# labels[0] = '$2^{-8}$'
# labels[1] = '$2^{-7}$'
# labels[2] = '$2^{-6}$'
# labels[3] = '$2^{-5}$'
# labels[4] = '$2^{-4}$'
# labels[5] = '$2^{-3}$'
# labels[6] = '$2^{-2}$'
# labels[7] = '$2^{-1}$'
# labels[8] = '$2^{0}$'
# labels[9] = '$1.5$'
# labels[10] = '$2^{1}$'
# labels[11] = '$2^{1}$'
ax.set_xticklabels(labels)
# ax.set_title("Paramater Study")
ax.set_ylabel("Average steps over 50 episodes (100 runs)", color='black')
ax.set_xlabel("[Actor-Critic] "+ r'$\alpha_{\theta}$' +" x number of tilings", color='black')

# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.06), shadow=True, ncol=1)
ax.legend(loc='best')
ax.set_ylim(100, 1000)
# fig.savefig("outputs/ac_alpha.pdf", bbox_inches='tight')
plt.show()
