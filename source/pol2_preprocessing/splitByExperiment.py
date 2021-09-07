# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/filtered.bed", 
                 sep="\t", header=None)

# %%
hasPol2 = df[3].str.split(".", expand=True)[1] == "POLR2A"
# %%
selected = df[hasPol2]
selected = selected[selected[4] > 3]
# %%
selected[[9,10]] = selected[3].str.split("_peak", expand=True) 
selected
# %%
splited = dict(([(k,x[np.arange(8)]) for k, x in selected.groupby(9)]))
# %%
peaksPerDataset = []
for f in splited.keys():
    if len(splited[f]) > 10000:
        splited[f].to_csv(f"/scratch/pdelangen/projet_these/data_clean/peaks/{f}.bed.gz", 
                          sep="\t", header=False, index=False)
    peaksPerDataset.append(len(splited[f]))

# %%
plt.figure()
plt.bar(np.arange(len(peaksPerDataset)), np.sort(peaksPerDataset))
plt.show()
plt.figure()
plt.hist(np.sort(peaksPerDataset), np.logspace(0, 5.2,50))
plt.xscale("log")
plt.vlines(1e4, 0, 100, "red", "dashed")
plt.show()
# %%
