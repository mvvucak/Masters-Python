import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import islice
import os

# np.set_printoptions(threshold=np.inf)
master_path = "E:\\Development Project\\Data\\GNPS Python Master\\Final Data.txt"
MAX_MASS = 1000
BIN_SIZE = 1

mol_all = np.loadtxt(master_path, dtype="U25")

print(mol_all.shape)
print(mol_all.dtype)

intensities = pd.DataFrame(0.0, index = mol_all[:100,0], columns=range(MAX_MASS//BIN_SIZE), dtype=float)

print(intensities.index)

print(list(intensities))

for row in mol_all[:100]:
    intensities.at[row[0], float(row[1])] = float(row[2])

print(intensities)

np_matrix = intensities.values

print(np.count_nonzero(np_matrix))
print(np_matrix.shape)
print(np_matrix.dtype)
print(np_matrix)
print(np.amax(np_matrix))
# mol_ids = np.loadtxt(master_path, usecols=0, dtype="U25")
# mol_data = np.loadtxt(master_path, usecols=[1, 2])
#
# mol_all = np.loadtxt(master_path, dtype="U25")
#
# print(mol_ids)
# print(len(mol_ids))
#
#
# mol_ids = np.unique(mol_ids)
#
# print(mol_ids)
# print(len(mol_ids))
#
# print(mol_data)
#
# intensities = pd.DataFrame(0.0, index = mol_ids, columns=range(MAX_MASS//BIN_SIZE), dtype=float)
#
# print(intensities[:1])
#
# for row in mol_all:
#     intensities.at[row[0], float(row[1])] = row[2]
#
# print(intensities.loc["CCMSLIB00000001548"])
# print(intensities)
#
# np_version = intensities.values
#
# print(np_version[0])
# print(len(np_version[0]))
# print(np_version.shape)
# print(np_version.dtype)
#
#
# plt.plot(np_version[0], color='g')
# plt.show()

