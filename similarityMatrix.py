import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

datapath = "E:\\Development Project\\Data\\GNPS" # Path to grab file names
similarities_path = "E:\\Development Project\\Data\\Pairwise Spectra Similarities\\matej_sim.csv"  # Similarities file

num_samples = 5770

headers = np.zeros(num_samples, dtype="U25")  # U25 defines maximum string length


def get_molecule_names():
    counter = 0
    for file in os.listdir(datapath):
        headers[counter] = file
        counter = counter + 1

get_molecule_names()
print(headers)
print(headers.shape)

similarities = pd.DataFrame(0, index=headers, columns=headers, dtype=float)


raw_similarities = pd.read_csv(similarities_path, header=None, usecols=[0,1,3])

print(raw_similarities.columns.values)
print(raw_similarities)
print(raw_similarities.values.shape)
print(similarities.columns.values[1])
print(similarities.columns.values[5769])
print(similarities.values.shape)
print(similarities)

print(similarities.at["CCMSLIB00000579427.ms", "CCMSLIB00000579423.ms"])

for row in raw_similarities.itertuples(index=True, name='Pandas'):
    mol_1 = row[1]
    mol_2 = row[2]
    value = row[3]
    similarities.at[mol_1, mol_2] = value
    # print(similarities.at[mol_1, mol_2])


print(similarities.at["CCMSLIB00000579427.ms", "CCMSLIB00000579423.ms"])

print(similarities)

# Plot rudimentary heatmap for bigger picture.
fig, ax = plt.subplots()
im = ax.imshow(similarities)

fig.tight_layout()
plt.show()

# Test to ensure that molecules aren't scoring high with themselves.
for h in headers:
    val = similarities.at[h,h]
    if val>0:
        print("Similarity for same molecule not 0")

# Turn into a numpy array
np_matrix = similarities.values

print(np_matrix.shape)
print(np_matrix.dtype)

print(np_matrix)

print(np.count_nonzero(np_matrix))
print(np_matrix.size)