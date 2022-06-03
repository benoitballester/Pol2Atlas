# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from lib import rnaseqFuncs
from statsmodels.stats.multitest import fdrcorrection
from settings import paths
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
except FileExistsError:
    pass
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
# %%
from lib.pyGREATglm import pyGREAT as pyGREATglm
enricherglm = pyGREATglm("/scratch/pdelangen/projet_these/data_clean/GO_files/hsapiens.GO:BP.name.gmt",
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/encode_counts/"
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/encode_total_rnaseq_annot_0.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in dlFiles:
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["Annotation"], sort=True)
# %%
# Deseq2 DE calculations
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
countsNz = allCounts[:, nzCounts]
# Scran normalization
sf = rnaseqFuncs.scranNorm(countsNz)
countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
# %%
for i in np.unique(ann):
    print(eq[i])
    labels = (ann == i).astype(int)
    res2 = rnaseqFuncs.mannWhitneyDE(countModel.residuals, sf, labels, order)
    sig = fdrcorrection(res2[0])[0] & (res2[1] > 0)
    res = pd.DataFrame(res2, columns=consensuses.index[nzCounts], index=["pval", "residual diff"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/res_{eq[i]}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/bed_{eq[i]}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/encode_rnaseq/DE/great_{eq[i]}.pdf")
# %%

# %%
