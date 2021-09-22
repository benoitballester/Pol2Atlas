# %%
import pandas as pd

origAnnot = pd.read_csv("/scratch/pdelangen/projet_these/data/peakMerge/annotcellLine.csv")
# %%
import os
allFiles = os.listdir("/scratch/pdelangen/projet_these/data_clean/peaksInterg/")
# %%
biotype = [i.split(".")[2].split("_")[0] for i in allFiles]
for i in range(len(biotype)):
    b = biotype[i]
    if b == "VcaP":
        biotype[i] = "VCaP"
    if b == "erythroblasts":
        biotype[i] = "erythroblast"
    if b == "HEK293T":
        biotype[i] = "HEK293t"
# %%
origAnnot.index = origAnnot["Cell_or_tissue"]
newFmt = origAnnot.loc[biotype][["Annotation"]]
newFmt["Cell_or_tissue"] = origAnnot["Cell_or_tissue"]
newFmt.index = allFiles
newFmt.index.name = "Sample"
newFmt
# %%
newFmt.to_csv("/scratch/pdelangen/projet_these/data_clean/annotPol2.tsv", sep="\t")
# %%
