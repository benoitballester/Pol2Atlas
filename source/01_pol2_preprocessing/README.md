# Pol II specific dataset preprocessing.
Transforms GENCODE gtf with 
Remove peaks with any overlap with blacklisted regions / genic regions +-1kb.
Select POLR2A only antibodies.
Split in one file per experiment.
Filter peaks with low q-value, remove experiments with small number of peaks.
