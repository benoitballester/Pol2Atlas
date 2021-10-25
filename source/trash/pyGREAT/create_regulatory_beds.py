# %%
import pandas as pd
import pandas as pd
import pyranges as pr
import numpy as np
sys.path.append("./")
from settings import params, paths

oboFilePath = paths.GOfolder + "/go_eq.obo"
outputFolder = paths.GOfolder
# %%
# Parse GO terms
allLines = []
termId = None
namespace = None
name = None
with open(oboFilePath) as f:
    for l in f.readlines():
        if l.startswith("[Term]"):
            if (not termId == None) and (not namespace == None) and (not name == None):
                allLines.append((termId, namespace, name))
            termId = None
            namespace = None
            name = None
        elif l.startswith("id"):
            termId = l.rstrip("\n").split(": ")[1]
        elif l.startswith("namespace"):
            namespace = l.rstrip("\n").split(": ")[1]
        elif l.startswith("name"):
            name = l.rstrip("\n").split(": ")[1]
df = pd.DataFrame(allLines)
df.columns = ["id", "namespace", "name"]
gb = df.groupby("namespace")
goClasses = [(x,gb.get_group(x)) for x in gb.groups]
for x, c in goClasses:
    c.to_csv(f"{outputFolder}GO_{x}.csv", sep="\t", index=None)
# %%
goAnnotation = pd.read_csv(paths.GOfolder + "/goa_human.gaf", sep="\t", skiprows=41, header=None)
# Remove NOT associations
goAnnotation = goAnnotation[np.logical_not(goAnnotation[3].str.startswith("NOT"))]
goAnnotation = goAnnotation[[2, 4]]
goAnnotation.dropna(inplace=True)
goFull = goAnnotation.merge(df, left_on=4, right_on="id")
goFull.drop(4, 1, inplace=True)
goFull.rename({2:"gene_name"}, axis=1, inplace=True)
goBP = goFull[goFull["namespace"] == "biological_process"]
# %%
gencode = pr.read_gtf(paths.gencode)
gencode = gencode.as_df()
transcripts = gencode[gencode["Feature"] == "gene"].copy()
del gencode
transcripts = transcripts[["Chromosome", "Start", "End", "gene_name", "Strand"]]
# %%
# Reverse positions on opposite strand
geneInList = np.isin(transcripts["gene_name"], np.unique(goAnnotation[2]), assume_unique=True)
reversedTx = transcripts.copy()[["Chromosome", "Start", "End", "gene_name"]][geneInList]
reversedTx["Start"] = transcripts["Start"].where(transcripts["Strand"] == "+", transcripts["End"])
reversedTx["End"] = transcripts["End"].where(transcripts["Strand"] == "+", transcripts["Start"])
upstream = 5000
downstream = 1000
distal = 1000000
reversedTx.sort_values(by=["Chromosome", "Start"], inplace=True)
regPm = reversedTx["Start"] - upstream * np.sign(reversedTx["End"]-reversedTx["Start"])
regPp = reversedTx["Start"] + downstream * np.sign(reversedTx["End"]-reversedTx["Start"])
gb = reversedTx.groupby("Chromosome")
perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
for c in perChr:
    inIdx = reversedTx["Chromosome"] == c
    previousReg = np.roll(regPp[inIdx], 1)
    previousReg[0] = 0
    previousReg[-1] = int(1e10)
    nextReg = np.roll(regPm[inIdx], -1)
    nextReg[-1] = int(1e10)
    extendedM = np.maximum(reversedTx["Start"][inIdx] - distal, np.minimum(previousReg, regPm[inIdx]))
    extendedP = np.minimum(reversedTx["Start"][inIdx] + distal, np.maximum(nextReg, regPp[inIdx]))
    reversedTx.loc[reversedTx["Chromosome"] == c, "Start"] = extendedM
    reversedTx.loc[reversedTx["Chromosome"] == c, "End"] = extendedP

# %%
fused = reversedTx.merge(goBP, on="gene_name")
fused.rename({4:"GO_Term"}, axis=1, inplace=True)
fused.to_csv(paths.GOfolder + "/gencode_GO.bed", sep="\t", index=None)
# %%
from scipy.sparse import coo_matrix, csr_matrix
geneFa, genes = pd.factorize(goBP["gene_name"])
goFa, gos = pd.factorize(goBP["id"])
data = np.ones_like(goFa, dtype="bool")
mat = coo_matrix((data, (geneFa, goFa)), shape=(len(genes), len(gos))).toarray().T
mat = pd.DataFrame(mat)
mat.columns = genes
mat.index = gos
# %%
from lib.utils import matrix_utils
clustersGO = matrix_utils.graphClustering(mat.values, "dice", r=0.8, restarts=10)
# %%
clust = 15
ids0 = gos[clustersGO == clust]
df.index = df["id"]
df.index.name="idx"
go0 = df.loc[ids0]
bestGO = np.sum(mat[clustersGO == clust], axis=1).idxmax()
print(df.loc[bestGO])
go0
# %%

# %%
