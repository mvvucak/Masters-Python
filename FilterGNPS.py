import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Test Binned"
filtered_datapath = "E:\\Development Project\\Data\\GNPS Python Test Filtered"
testpath = "E:\\Development Project\\Data\\GNPS Python Test Binned\\CCMSLIB00000001548 Binned.txt"


def normalize_and_filter(spectrum, max_value = 1000, min_percent=0.005):
    min_value = max_value * min_percent
    # print(min_value)
    max_intensity = np.amax(spectrum)  # Find max intensity
    spectrum = spectrum / max_intensity  # Normalize
    spectrum = spectrum * max_value  # Scale all values
    filtered = np.where(spectrum < min_value, 0, spectrum)  # Set values below threshold to 0.
    # print(np.sum(spectrum > 0))
    # print(np.sum(filtered > 0))
    # print(np.amin(spectrum[np.nonzero(spectrum)]))
    # print(np.amin(filtered[np.nonzero(filtered)]))
    return filtered

def process_files():
    for file in os.listdir(datapath):
        filepath = os.path.join(datapath, file)
        data = np.loadtxt(filepath, np.float32)
        data[:, 1] = normalize_and_filter(data[:, 1])
        # print(data.shape)
        filtered_filename = file.split("Binned")[0] + "Filtered.txt"
        filtered_filepath = os.path.join(filtered_datapath, filtered_filename)
        np.savetxt(filtered_filepath, data, fmt="%d %f")

process_files()

# file = np.loadtxt(testpath, np.float32)
#
# normalize_and_filter(file[:,1])
