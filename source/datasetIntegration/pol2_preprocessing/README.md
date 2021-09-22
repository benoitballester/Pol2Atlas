# Pol II specific dataset preprocessing.
Remove peaks with any overlap with blacklisted regions / genic regions +-1kb.
Select POLR2A only antibodies.
Split in one file per experiment.
Threshold at 1e-5 MACS2 qval
# Transforming some files
-Repeats
-Gencode GTF