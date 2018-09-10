import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

names_datapath = "E:\\Development Project\\Data\\GNPS" # Path to grab file names
similarities_path = "E:\\Development Project\\Data\\Pairwise Spectra Similarities\\matej_sim.csv"  # Similarities file

num_samples = 5770
batch_size = 500

headers = np.zeros(num_samples, dtype="U25")
# Get file names to use as headers.
for i, file in enumerate(os.listdir(names_datapath)):
    headers[i] = file
# Create empy dataframe to hold matrix, read in similarity values.
similarities = pd.DataFrame(0, index=headers, columns=headers, dtype=float)
raw_similarities = pd.read_csv(similarities_path, header=None, usecols=[0, 1, 3])
# Insert similarity value for molecule pair in correct cell.
for row in raw_similarities.itertuples(index=True, name='Pandas'):
    mol_1 = row[1]
    mol_2 = row[2]
    value = row[3]
    similarities.at[mol_1, mol_2] = value

# Turn into a 2D numpy array
np_form = similarities.values

# Find rows where all values are 0, change to 1/num_samples to avoid /0
# sums = np.sum(np_form, axis=1)
# print(np.where(sums == 0))
# np_form[np.where(sums == 0)] = 1 / num_samples

print(np.amin(np_form))

# Turn into a 3D numpy array, representing batches

num_batches = int(num_samples / batch_size)

np_matrix = np.zeros((num_batches, batch_size, batch_size))

for i, start in enumerate(range(0, num_samples - batch_size + 1, batch_size)):
    np_matrix[i] = np_form[start:start + batch_size, start:start + batch_size]
    sums = np.sum(np_matrix[i], axis=1)
    np_matrix[i][np.where(sums == 0)] = 1 / batch_size
    np_matrix[i] = 0.5 * (np_matrix[i] + np_matrix[i].T)  # Make symmetric?
    np_matrix[i] = np_matrix[i] / np_matrix[i].sum(axis=1)[:, None]  # Normalize by row.


# np_matrix = 0.5 * (np_matrix + np_matrix.T) # Make symmetric?

# print(sums)
# np_matrix = np_matrix/np_matrix.sum(axis=1)[:, None] # Normalize by row.
# sums = np.sum(np_matrix, axis=1)
# print(np.where(sums<0.99)) # Verify all rows sum up to 1.

#print(np_matrix[0])