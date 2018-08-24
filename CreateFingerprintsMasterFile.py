import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\Fingerprint Bitmaps 2"
final_path = "E:\\Development Project\\Data\\GNPS Python Master\\Final Fingerprints.txt"

all_files = os.listdir(datapath)

with open(final_path, 'w') as f:  # The master file to write to
    for file in all_files:  # For each filtered file
        file_path = os.path.join(datapath, file)
        with open(file_path, 'r') as d:  # Open and read the file
            print(file)
            mol_id = file.split('.')[0]
            lines = list(islice(d, 0, None))
            for line_number, line in enumerate(lines):
                bit = int(line)
                if bit != 0:
                    f.write(mol_id + " ")
                    f.write(str(line_number) + " ")
                    f.write(str(bit) + "\n")

