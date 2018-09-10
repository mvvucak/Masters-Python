import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Binned"
filtered_datapath = "E:\\Development Project\\Data\\GNPS Python Filtered"

double_filtered_datapath = "E:\\Development Project\\Data\\GNPS Python Double Filtered"

fragment_id_datapath = "E:\\Development Project\\Data\\GNPS Python With Fragments"

mass_path = "E:\\Development Project\\Data\\Fragment Masses.txt"

# Load fragment data for IDing
dtype = [('fragment', 'U25'), ('mass', int)]
fragments = np.loadtxt(mass_path, dtype=dtype, delimiter=',')


def normalize_and_filter(spectrum, max_value=1000, min_percent=0.005):
    min_value = max_value * min_percent
    # print(min_value)
    max_intensity = np.amax(spectrum)  # Find max intensity
    spectrum = spectrum / max_intensity  # Normalize to 0-1
    spectrum = spectrum * max_value  # Scale all values
    filtered = np.where(spectrum < min_value, 0, spectrum)  # Set values below threshold to 0.
    return filtered


def top_six_filter(spectrum):
    filtered_spectrum = np.zeros(spectrum.shape, float)
    for i in range(len(spectrum)):  # For each mass bin
        low_end = 0
        if i < 50:
            low_end = i  # If there are fewer than 50 bins behind current windows, only go back to index 0.
        if i >= 50:
            low_end = 50  # Else, go back 50 indices
        window_comparison = np.less(spectrum[i], spectrum[i-low_end:(i+50)])  # Compare value to all bins in 100Da range
        if np.sum(window_comparison) < 7:  # If value is among top 6 in 100Da range, add it to filtered array.
            filtered_spectrum[i] = spectrum[i]
    return filtered_spectrum


def scan_for_fragments(spectrum):
    intensity_only = spectrum[:,1]

    for i, fragment in enumerate(fragments):
        f_mass = fragment['mass']
        # Slice intensities twice so bins to be compared share index.
        spec = intensity_only[:(intensity_only.size - f_mass)]
        comp_spec = intensity_only[f_mass:]
        pairs = np.where(np.logical_and(spec > 0, comp_spec > 0))  # Compare bins separated by residue mass
        f_intensity = 0  # Default value for fragment presence
        if pairs[0].size > 0:
            # Sum up intensities of fragments separated by fragment mass and take highest.
            f_intensity = np.amax(spec[pairs]*comp_spec[pairs])
            #if f_intensity > 1000:
                #f_intensity = 1000
        # Append fragment information to spectrum
        new_row = np.array([intensity_only.size+i+1, f_intensity], dtype=float)
        spectrum = np.vstack([spectrum, new_row])
    return spectrum


def process_files():
    for file in os.listdir(datapath):
        print(file)
        filepath = os.path.join(datapath, file)
        data = np.loadtxt(filepath, np.float32)
        # Scale and filter all below threshold. Column 1 contains intensities.
        data[:, 1] = normalize_and_filter(data[:, 1])
        # Save to new file.
        filtered_filename = file.split("Binned")[0] + "Filtered.txt"
        filtered_filepath = os.path.join(filtered_datapath, filtered_filename)
        np.savetxt(filtered_filepath, data, fmt="%d %f")

        # Top 6 Filter
        data[:, 1] = top_six_filter(data[:, 1])

        double_filtered_filename = file.split("Binned")[0] + "Filtered Twice.txt"
        double_filtered_filepath = os.path.join(double_filtered_datapath, double_filtered_filename)
        np.savetxt(double_filtered_filepath, data, fmt="%d %f")

        # Identify potential fragments
        data = scan_for_fragments(data)

        fragment_id_filename = file.split("Binned")[0] + " With Fragments.txt"
        fragment_id_filepath = os.path.join(fragment_id_datapath, fragment_id_filename)
        np.savetxt(fragment_id_filepath, data, fmt="%d %f")


process_files()

# file = np.loadtxt(testpath, np.float32)
#
# normalize_and_filter(file[:,1])
