from itertools import islice
import os

datapath = "E:\\Development Project\\Data\\Fingerprint Bitmaps 2\\Fingerprint Legend"
final_path = "E:\\Development Project\\Data\\GNPS Python Master\\Fingerprint Legend.txt"

with open(final_path, 'w') as dest:
    with open(datapath, 'r') as source:
        lines = list(islice(source, 0, None))
        for line in lines:
            print(line)
            substructure = line.split('\t', maxsplit=2)[1]
            print(substructure)
            dest.write(substructure + "\n")

fingerprint_legend = []
with open(final_path, 'r') as f:
    lines = list(islice(f, 0, None))
    for line in lines:
        fingerprint_legend.append(line)

print(fingerprint_legend)