import numpy as np
import os
import pandas as pd
import scipy
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt

datapath = "E:\\Development Project\\Data\\Kernel Experiments"

variables = ["Convolution Only", "30"]
num_runs = 10

var_scores = []
var_scores_08 = []

for variable in variables:
    auc_scores = []
    auc_scores_08 = []
    for i in range(num_runs):
        filename = variable + " " + str(i) + ".txt"
        filepath = os.path.join(datapath, filename)
        stats = np.loadtxt(filepath, dtype=float)
        auc_above_07 = len(np.where(stats[:, 2] > 0.7)[0])
        auc_above_08 = len(np.where(stats[:, 2] > 0.8)[0])
        auc_scores.append(auc_above_07)
        auc_scores_08.append(auc_above_08)
    var_scores.append(auc_scores)
    var_scores_08.append(auc_scores_08)

tstat, pval = scipy.stats.ttest_rel(var_scores[0], var_scores[1])

print("For 0.7: t stat:" + str(tstat) + "  p:" + str(pval))

tstat, pval = scipy.stats.ttest_rel(var_scores_08[0], var_scores_08[1])

print("For 0.8: t stat:" + str(tstat) + "  p:" + str(pval))
