import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import islice
import os

# np.set_printoptions(threshold=np.inf)
master_path = "E:\\Development Project\\Data\\GNPS Python Master\\Final Fingerprints.txt"
BITS = 320  # Total number of bits in fingerprint

fp_all = np.loadtxt(master_path, dtype="U25")
fp_ids = np.unique(fp_all[:, 0])

print(fp_all.shape)
print(fp_all.dtype)

fingerprints = pd.DataFrame(0, index = fp_ids, columns=range(BITS), dtype=int)

print(fingerprints.index)

print(list(fingerprints))

for row in fp_all:
    fingerprints.at[row[0], int(row[1])] = int(row[2])

print(fingerprints)

np_matrix = fingerprints.values

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

