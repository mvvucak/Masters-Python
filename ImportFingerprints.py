import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from itertools import islice

fingerprint_path = "E:\\Development Project\\Data\\Fingerprint Bitmaps 2"

num_samples = 5770  # As below
num_features = 320  # Can turn this into a function that goes through the files instead of hard-coded.

substructure_fingerprints = np.zeros([num_features, 0], int)
filenames = []

counter = 0
erroneous = []
for file in os.listdir(fingerprint_path):
    filepath = os.path.join(fingerprint_path, file)
    filenames.append(file)
    fingerprint = np.loadtxt(filepath, int)
    # if(fingerprint.shape[0]>320):
    #     print(file)
    #     print(fingerprint.shape)
    #     counter = counter + 1
    #     erroneous.append(fingerprint.shape[0])
    fingerprint = fingerprint.reshape(-1, 1)
    # fingerprint = np.zeros([num_features, 1], int)
    # with open(filepath, 'r') as f:
    #     counter = 0
    #     lines = list(islice(f, 0, None))
    #     for line in lines:
    #         fingerprint[counter] = int(line)
    #         counter = counter + 1
    substructure_fingerprints = np.concatenate([substructure_fingerprints, fingerprint], axis=1)


# print(counter)
# print(erroneous)
# print(min(erroneous))
print(substructure_fingerprints.shape)
print(np.amax(substructure_fingerprints))
print(filenames[1])
print(substructure_fingerprints[:, 1])
zero_only_bits = []
for i in range(319):
    if(np.amax(substructure_fingerprints[i])) == 0:
        zero_only_bits.append(i)
print(zero_only_bits)
print(len(zero_only_bits))


