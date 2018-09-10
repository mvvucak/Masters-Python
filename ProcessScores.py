import numpy as np
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 18})

datapath = "E:\\Development Project\\Data\\Kernel Experiments 100"

sizes = ["Base", "3",  "16", "30", "60"]
names = ["Base", "3",  "16", "30", "60"]

num_runs = 10

auc_means = []
auc_stds = []
auc_means_08 = []
auc_stds_08 = []
perm_means = []
perm_stds = []

for size in sizes:
    auc_scores = []
    auc_scores_08 = []
    perm_scores = []
    for i in range(num_runs):
        filename= size + " " + str(i) + ".txt"
        filepath = os.path.join(datapath, filename)
        stats = np.loadtxt(filepath, dtype=float)
        auc_above_07 = len(np.where(stats[:,2] > 0.7)[0])
        auc_above_08 = len(np.where(stats[:,2] > 0.8)[0])
        auc_scores.append(auc_above_07)
        auc_scores_08.append(auc_above_08)
        perm_below_005 = len(np.where(stats[:, 3] < 0.05)[0])
        perm_scores.append(perm_below_005)
    auc_means.append(np.mean(auc_scores))
    auc_stds.append(np.std(auc_scores))
    auc_means_08.append(np.mean(auc_scores_08))
    auc_stds_08.append(np.std(auc_scores_08))
    perm_means.append(np.mean(perm_scores))
    perm_stds.append(np.std(perm_scores))

print(auc_means)
print(auc_stds)

fig, ax = plt.subplots()

index = np.arange(len(sizes))

bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

scores = ax.bar(index, auc_means, bar_width, alpha=opacity, color='b', yerr = auc_stds, error_kw = error_config, label = "AUC Above 0.7")
scores2 = ax.bar(index+bar_width, auc_means_08, bar_width, alpha=opacity, color='r', yerr = auc_stds_08, error_kw = error_config, label = "AUC Above 0.8")

ax.set_xlabel('Kernel Size')
ax.set_ylabel('Number of Substructures')
ax.set_title('Substructures Above AUC Threshold by Kernel Size')
ax.set_xticks(index+bar_width/2)
ax.set_xticklabels(names)
ax.legend(loc='upper left')

plt.show()
#
fig, ax = plt.subplots()

# scores = ax.bar(index, perm_means, bar_width, alpha=opacity, color='g', yerr = perm_stds, error_kw = error_config, label = "Perm % Below 5%")
#
# ax.set_xlabel('Kernel Size')
# ax.set_ylabel('Number of Substructures with Perm % < 5')
# ax.set_title('Substructures Above Below Permutation % Threshold by Kernel Size')
# ax.set_xticks(index)
# ax.set_xticklabels(names)
# ax.legend()
#
# plt.show()





