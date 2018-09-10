import numpy as np
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 16})

datapath = "E:\\Development Project\\Data\\GNPS Python Binned"
datapath_norm = "E:\\Development Project\\Data\\GNPS Python Filtered"
datapath_six = "E:\\Development Project\\Data\\GNPS Python Double Filtered"

paths = [datapath, datapath_norm, datapath_six]

file = "CCMSLIB00000001548"

binned_file = file + " Binned.txt"
normalized_file = file + " Filtered.txt"
filtered_file = file + " Filtered Twice.txt"

files = [binned_file, normalized_file, filtered_file]
print(files)

fig, ax = plt.subplots()

ax.plot([60, 60], [0, 78], color='g')
ax.plot([178, 178], [0, 459], color='g')
ax.plot([212, 212], [0, 578], color='g')
ax.plot([489, 489], [0, 603], color='g')
ax.plot([549, 549], [0, 1000], color='g')
ax.plot([999,999], [0,0], color='g')
ax.set_xlabel("Mass Bin(Da)")
ax.set_ylabel("Relative Abundance")
ax.ticklabel_format(useMathText=True)
ax.set_ylim(ymin=0)
plt.show()


for i, path in enumerate(paths):
    filepath = os.path.join(path, files[i])
    print(filepath)
    spectrum = np.loadtxt(filepath)[:,1]
    fig, ax = plt.subplots()
    for i, d in enumerate(spectrum):
        if d>0.1:
            ax.plot([i,i],[0, d], color='g')
    ax.plot([999,999], [0,0])
    ax.set_xlabel("Mass Bin(Da)")
    ax.set_ylabel("Relative Abundance")
    ax.ticklabel_format(useMathText=True)
    ax.set_ylim(ymin=0)
    plt.show()

