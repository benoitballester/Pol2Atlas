# %%
genesPerAnnot = dict()
goMap = dict()
allGenes = set()
with open("/scratch/pdelangen/projet_these/data_clean/GO_files/hsapiens.GO:BP.name.gmt") as f:
    for l in f:
        vals = l.rstrip("\n").split("\t")
        genesPerAnnot[vals[0]] = vals[2:]
        allGenes |= set(vals[2:])
        goMap[vals[0]] = vals[1]

import pandas as pd
import numpy as np
mat = pd.DataFrame(columns=allGenes, dtype="int8", index=genesPerAnnot.keys())
for ann in genesPerAnnot.keys():
    mat.loc[ann] = 0
    mat.loc[ann][genesPerAnnot[ann]] = 1
# %%
