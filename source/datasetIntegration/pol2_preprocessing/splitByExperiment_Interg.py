# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("./")
from settings import params, paths
df = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredInterg.bed", 
                 sep="\t", header=None)
logQvalCutoff = 5
peakCountCutoff = 100
# %%
hasPol2 = df[3].str.split(".", expand=True)[1] == "POLR2A"
# %%
selected = df[hasPol2]
plt.figure()
plt.hist(selected[4], np.arange(0,100))
plt.vlines(logQvalCutoff, plt.ylim()[0], plt.ylim()[1], "red", "dashed")
plt.xlabel("-log10(q-value) per peak")
plt.ylabel("Number of peaks")
plt.savefig(paths.outputDir + "descriptivePlots/distribQval.pdf")
plt.show()
selected = selected[selected[4] > logQvalCutoff]

# %%
selected[[9,10]] = selected[3].str.split("_peak", expand=True) 
selected
# %%
splited = dict(([(k,x[np.arange(8)]) for k, x in selected.groupby(9)]))
# %%
peaksPerDataset = []
for f in splited.keys():
    if len(splited[f]) > peakCountCutoff:
        splited[f].to_csv(f"/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/peaksInterg/{f}.bed.gz", 
                        sep="\t", header=False, index=False)
        a=0
    peaksPerDataset.append(len(splited[f]))
peaksPerDataset = np.array(peaksPerDataset)
# %%
plt.figure()
plt.hist(np.sort(peaksPerDataset), np.logspace(0, 5.2,50))
plt.xscale("log")
plt.vlines(peakCountCutoff, 0, 100, "red", "dashed")
plt.xlabel("Intergenic peak count per dataset")
plt.ylabel("Number of datasets")
plt.savefig(paths.outputDir + "descriptivePlots/distribIntergPeaksPerDataset.pdf")
plt.show()
# %%
