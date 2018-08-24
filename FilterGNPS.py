import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Binned"
filtered_datapath = "E:\\Development Project\\Data\\GNPS Python Filtered"
testpath = "E:\\Development Project\\Data\\GNPS Python Test Binned\\CCMSLIB00000001548 Binned.txt"
double_filtered_datapath = "E:\\Development Project\\Data\\GNPS Python Double Filtered"


def normalize_and_filter(spectrum, max_value = 1000, min_percent=0.005):
    min_value = max_value * min_percent
    # print(min_value)
    max_intensity = np.amax(spectrum)  # Find max intensity
    spectrum = spectrum / max_intensity  # Normalize to 0-1
    spectrum = spectrum * max_value  # Scale all values
    filtered = np.where(spectrum < min_value, 0, spectrum)  # Set values below threshold to 0.
    # print(np.sum(spectrum > 0))
    # print(np.sum(filtered > 0))
    # print(np.amin(spectrum[np.nonzero(spectrum)]))
    # print(np.amin(filtered[np.nonzero(filtered)]))
    return filtered

def top_six_filter(spectrum):
    filtered_spectrum = np.zeros(spectrum.shape, float)
    for i in range(len(spectrum)):
        lowend = 0
        if i < 50:
            lowend = i  # If there are fewer than 50 bins behind current windows, only go back to index 0.
        if i >= 50:
            lowend = 50  # Else, go back 50 indices
        window_comparison = np.less(spectrum[i], spectrum[i-lowend:(i+50)])  # Compare current value to values for all bins in 100Da range
        if np.sum(window_comparison)<7:  # If value is among top 6 in 100Da range, add it to filtered array.
            filtered_spectrum[i] = spectrum[i]
    return filtered_spectrum


def process_files():
    for file in os.listdir(datapath):
        filepath = os.path.join(datapath, file)
        data = np.loadtxt(filepath, np.float32)
        # Scale and filter all below threshold
        data[:, 1] = normalize_and_filter(data[:, 1])

        filtered_filename = file.split("Binned")[0] + "Filtered.txt"
        filtered_filepath = os.path.join(filtered_datapath, filtered_filename)
        np.savetxt(filtered_filepath, data, fmt="%d %f")

        # Top 6 Filter
        data[:, 1] = top_six_filter(data[:, 1])

        double_filtered_filename = file.split("Binned")[0] + "Filtered Twice.txt"
        double_filtered_filepath = os.path.join(double_filtered_datapath, double_filtered_filename)
        np.savetxt(double_filtered_filepath, data, fmt="%d %f")


process_files()

# file = np.loadtxt(testpath, np.float32)
#
# normalize_and_filter(file[:,1])
