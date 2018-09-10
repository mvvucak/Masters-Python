import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Binned"
mass_path = "E:\\Development Project\\Data\\Fragment Masses.txt"

dtype = [('fragment', 'U25'), ('mass', np.float32)]
dtype_binned = [('fragment', 'U25'), ('mass', int)]

fragments = np.loadtxt(mass_path, dtype=dtype, delimiter=',')

fragments = fragments.astype(dtype_binned)

def scan_for_fragments(spectrum):
    intensity_only = spectrum[:,1]
    for i, fragment in enumerate(fragments):
        f_mass = fragment['mass']
        spec = intensity_only[:(intensity_only.size - f_mass)]
        comp_spec = intensity_only[f_mass:]  # Offset intensities, for comparison
        pairs = np.where(np.logical_and(spec > 0, comp_spec > 0))  # Compare bins separated by residue mass
        f_intensity = 0
        if pairs[0].size > 0:
            f_intensity = np.amax(spec[pairs]+comp_spec[pairs])
        new_row = np.array([intensity_only.size+i+1, f_intensity], dtype=float)
        spectrum = np.vstack([spectrum, new_row])  # Append mass shift information to spectrum
    print(spectrum.shape)
    return spectrum


def process_files():
    for file in os.listdir(datapath):
        filepath = os.path.join(datapath, file)
        data = np.loadtxt(filepath, np.float32)
        data = scan_for_fragments(data)


process_files()