# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
import pyranges as pr
eqtlPath = paths.gtexData + "eQTL/GTEx_Analysis_v8_eQTL/"
eqtlFiles = os.listdir(eqtlPath)
eqtlFiles = [f for f in eqtlFiles if f.endswith("signif_variant_gene_pairs.txt.gz")]
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None, usecols=[0,1,2,3])
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
# %%
# Read per tissue pol2 markers
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
indices = dict()
for f in filesEncode:
    name = f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals
# Build matrix
from scipy.sparse import csr_matrix
rows = []
cols = []
data = []
for i, k in enumerate(indices.keys()):
    idx = list(indices[k])
    cols += idx
    rows += [i]*len(idx)
    data += [True]*len(idx)
mat = csr_matrix((data, (rows, cols)), shape=(len(indices), np.max(cols)+1), dtype=bool)
mat = pd.DataFrame(mat.todense(), index=indices.keys())
# Throw away non-markers and low specificity markers
kept = (mat.sum(axis=0) <= 2).values & (mat.sum(axis=0) != 0).values
consensuses = consensuses.iloc[kept]
mat = mat.iloc[:, kept]

# %%
'''
tissue = eqtlFiles[7].split(".")[0].split("_")[0]
df = pd.read_csv(eqtlPath + eqtlFiles[7], sep="\t", usecols=[0,1])
dfBed = df["variant_id"].str.split("_", expand=True).iloc[:, :2]
dfBed.columns = ["Chromosome", "Start"]
dfBed["Start"] = dfBed["Start"].astype("int")
dfBed["End"] = dfBed["Start"] + 1
dfBed["Name"] = df["gene_id"]
dfBed = pr.PyRanges(dfBed)
tissueMarkers = pr.PyRanges(consensuses[mat.loc["Heart"]])
from scipy.stats import hypergeom
k = len(tissueMarkers.overlap(dfBed))
n = len(pr.PyRanges(consensuses).overlap(dfBed))
M = len(consensuses)
N = len(tissueMarkers)
p = hypergeom(M, n, N).sf(k-1)
fc = (k/N)/(n/M)
print(tissue, fc, p)
'''
# %%
# Read eqtl files and store eQTL-gene association vs observed in tissue in a binary matrix
tissueEqtls = dict()
allEqtls = np.array([])
tissues = set()
for f in eqtlFiles:
    tissue = f.split(".")[0]
    df = pd.read_csv(eqtlPath + f, sep="\t", usecols=[0,1])
    eqtls = (df["variant_id"] + "_" + df["gene_id"]).values
    tissueEqtls[tissue] = eqtls
    allEqtls = np.unique(np.concatenate([eqtls, allEqtls]))
    tissues.add(tissue)
eqtlDf = pd.DataFrame(np.zeros((len(allEqtls), len(tissues)), "bool"), index=allEqtls, columns=tissues)
for i in tissues:
    eqtlDf.loc[tissueEqtls[i], i] = True

# %%
# Trim low-specificity eQTL-gene associations
highSpec = eqtlDf.sum(axis=1) <= 2
eqtlPos = pd.Series(allEqtls).str.split("_", expand=True)[[0,1,5]]
eqtlPos.columns = [["Chromosome", "Start", "Name"]]
eqtlPos["Start"] = eqtlPos["Start"].astype("int")
eqtlPos["End"] = eqtlPos["Start"] + 1

# %%
# Compute enrichments of per tissue eqtl in per tissue Pol II markers
fcDf = pd.DataFrame(index=tissues, columns=mat.index)
pvalDf = pd.DataFrame(index=tissues, columns=mat.index)
for i in tissues:
    eqtlPR = pr.PyRanges(eqtlPos.iloc[highSpec & eqtlDf[]])
    for j in mat.index:
