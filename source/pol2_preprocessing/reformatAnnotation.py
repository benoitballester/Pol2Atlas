# %%
import pandas as pd

origAnnot = pd.read_csv("/scratch/pdelangen/projet_these/data/peakMerge/annotcellLine.csv")
# %%
import os
allFiles = os.listdir("/scratch/pdelangen/projet_these/data_clean/peaks/")
# %%
biotype = [i.split(".")[2].split("_")[0] for i in allFiles]
for i in range(len(biotype)):
    b = biotype[i]
    if b == "VcaP":
        biotype[i] = "VCaP"
# %%
origAnnot.index = origAnnot["Cell_or_tissue"]
newFmt = origAnnot.loc[biotype][["Annotation"]]
newFmt["Sample"] = allFiles
newFmt.to_csv("/scratch/pdelangen/projet_these/data_clean/annotPol2.tsv", sep="\t")
# %%
