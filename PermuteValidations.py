import numpy as np

perm_file_path = "E:\\Development Project\\Data\\Validation Spit Permutations.txt"
# Create and store 10 randomly permuted indices for 5770 spectra

att = np.arange(5770)

att = np.random.permutation(att)

index_permutations = np.zeros((5770, 0), dtype=int)
# Add each permutation to a numpy array of indices
for i in range(10):
    perm = np.random.permutation(np.arange(5770, dtype=int))
    index_permutations = np.column_stack((index_permutations, perm))

# Verify numpy array has correct shape (should be 5770 for each column)
for i in range(10):
    print(np.unique(index_permutations[:,i]).size)

# Save index permutations to file
np.savetxt(perm_file_path, index_permutations, delimiter=',', fmt='%d')

