import numpy as np
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 20})

datapath = "E:\\Development Project\\Data\\Kernel Experiments 100\\"

variables = ["Base", "3", "16", "30", "60"]

filepath_one = datapath + variables[0] + " 0.txt"
stats_one = np.loadtxt(filepath_one, dtype=float)
filter = np.where(stats_one[:,2]>0.5)
stats_one = stats_one[filter]

filepath_two = datapath + variables[1] + " 0.txt"
stats_two = np.loadtxt(filepath_two, dtype=float)
filter = np.where(stats_two[:,2]>0.5)
stats_two = stats_two[filter]

filepath_three = datapath + variables[2] + " 0.txt"
stats_three = np.loadtxt(filepath_three, dtype=float)
filter = np.where(stats_three[:,2]>0.5)
stats_three = stats_three[filter]

filepath_four = datapath + variables[3] + " 0.txt"
stats_four = np.loadtxt(filepath_four, dtype=float)
filter = np.where(stats_four[:,2]>0.5)
stats_four = stats_four[filter]

filepath_five = datapath + variables[4] + " 0.txt"
stats_five = np.loadtxt(filepath_five, dtype=float)
filter = np.where(stats_five[:,2]>0.5)
stats_five = stats_five[filter]

fig, ax = plt.subplots(figsize=(8,4))

n_bins = 1000

n, bins, patches = ax.hist(stats_one[:,2], n_bins, normed=1, histtype ='step', cumulative=-1, label='Basic')

ax.hist(stats_two[:,2], n_bins, normed=1, histtype='step', cumulative=-1, label='3')

ax.hist(stats_three[:,2], n_bins, normed=1, histtype='step', cumulative=-1, label='16')

ax.hist(stats_four[:,2], n_bins, normed=1, histtype='step', cumulative=-1, label='30')

ax.hist(stats_five[:,2], n_bins, normed=1, histtype='step', cumulative=-1, label='60')

ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative AUC Scores')
ax.set_xlabel('AUC Score')
ax.set_ylabel('Proportion of Substructures')

plt.show()