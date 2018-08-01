import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Test"
binned_datapath = "E:\\Development Project\\Data\\GNPS Python Test Binned"
testpath = "E:\\Development Project\\Data\\GNPS\\CCMSLIB00000004552.ms"



def find_max_mass():
    mass_list = []
    for file in os.listdir(datapath):
        filepath = os.path.join(datapath, file)
        with open(filepath, 'r') as f:
            unsplit_values = list(islice(f, 9, None))
            for line in unsplit_values:
                if ' ' in line:
                    next_mass = int(round(float(line.split()[0])))
                    mass_list.append(next_mass)
                    # print(line.split()[0])
                    # print(int(round(float(line.split()[0]))))
                    # print(line.split()[1])
    return max(mass_list)

def write_binned_files():
    max_bins = find_max_mass() + 1
    for file in os.listdir(datapath):
        filepath = os.path.join(datapath, file)
        binned_values = np.zeros(max_bins)
        with open(filepath, 'r') as f:
            filename = f.name

            unsplit_lines = list(islice(f, 9, None))
            for line in unsplit_lines:
                if ' ' in line:  # Only lines with mass and intensity values have a space. Ignores label/blank lines
                    split_line = line.split()
                    mass = int(round(float(split_line[0])))  # Serves as index for binned values array
                    intensity = float(split_line[1])
                    #print ("The intensity for mass " + str(mass) + " is " + str(binned_values[mass]))
                    binned_values[mass] = binned_values[mass] + intensity
                    #print("The intensity for mass " + str(mass) + " is " + str(binned_values[mass]))
        binned_filename = file.split(".")[0] + " Binned.txt"
        binned_filepath = os.path.join(binned_datapath, binned_filename)
        with open(binned_filepath, 'w') as f:
            for intensity in binned_values:
                f.write(str(intensity) + "\n")




write_binned_files()