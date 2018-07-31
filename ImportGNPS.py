import numpy as np
import matplotlib as plt
from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\GNPS"
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
                 # print(max_mass)
                 # print(line.split()[0])
                 # print(int(round(float(line.split()[0]))))
                 # print(line.split()[1])

    return max(mass_list)

max_mass = find_max_mass()
print(max_mass)