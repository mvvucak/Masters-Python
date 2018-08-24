import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS Python Double Filtered"
final_path = "E:\\Development Project\\Data\\GNPS Python Master\\Final Data.txt"

all_files = os.listdir(datapath)

with open(final_path, 'w') as f:  # The master file to write to
    for file in all_files:  # For each filtered file
        file_path = os.path.join(datapath, file)
        with open(file_path, 'r') as d:  # Open and read the file
            print(file)
            mol_id = file.split()[0]
            lines = list(islice(d, 0, None))
            for line in lines:
                split_line = line.split()
                mass = split_line[0]
                intensity = float(split_line[1])
                if intensity != 0.0:
                    f.write(mol_id + " ")
                    f.write(mass + " ")
                    f.write(str(intensity) + "\n")

