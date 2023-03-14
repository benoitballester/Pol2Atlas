# %%
import os
import sys
print(os.sched_getaffinity(0))
print(len(os.sched_getaffinity(0)))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kaplanmeier as km
sys.path.append("./")
import warnings

import lifelines as ll
import rpy2.robjects as ro
import seaborn as sns
from joblib import Parallel, delayed
from lib import rnaseqFuncs
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import chi2, mannwhitneyu, rankdata, ttest_ind
from settings import params, paths
from statsmodels.stats.multitest import fdrcorrection
survival = importr("survival")
scran = importr("scran")
maxstat = importr("maxstat")
np.random.seed(42)
try:
    os.mkdir(paths.outputDir + "rnaseq/Survival/Kaplan150/")
except FileExistsError:
    pass
# %%
# Setup go enrichments and chromosome info
chrFile = pd.read_csv(paths.genomeFile, sep="\t", index_col=0, header=None)
sortedIdx = ["chr1", 'chr2','chr3','chr4','chr5','chr6',
              'chr7','chr8','chr9', 'chr10', 'chr11','chr12','chr13','chr14','chr15','chr16','chr17',
              'chr18','chr19','chr20','chr21','chr22','chrX','chrY']
chrFile = chrFile.loc[sortedIdx]
from lib.pyGREATglm import pyGREAT

enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
import pyranges as pr

# %%
# Load dataset annotations
allAnnots = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
try:
    os.mkdir(paths.outputDir + "rnaseq/Survival/")
except FileExistsError:
    pass
progPerCancer = pd.DataFrame()
statsCase = dict()
studiedConsensusesCase = dict()
nzPerCancer = dict()
cases = allAnnots["project_id"].unique()
# %%
multiProg = pd.read_csv(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed", header=None, sep="\t")
# %%
# For each cancer
for case in cases:
    print(case)
    # Select only relevant files and annotations
    annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv",
                            sep="\t", index_col=0)
    annotation = annotation[annotation["project_id"] == case]
    annotation = annotation[np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")]
    dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
    dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
    ids = np.array([f.split(".")[0] for f in dlFiles])
    inAnnot = np.isin(ids, annotation.index)
    ids = ids[inAnnot]
    dlFiles = np.array(dlFiles)[inAnnot]
    annotation = annotation.loc[ids]
    # Read survival information
    survived = (annotation["vital_status"] == "Alive").values
    timeToEvent = annotation["days_to_last_follow_up"].where(survived, annotation["days_to_death"])
    # Drop rows with missing survival information
    toDrop = timeToEvent.index[timeToEvent == "'--"]
    boolIndexing = np.logical_not(np.isin(timeToEvent.index, toDrop))
    timeToEvent = timeToEvent.drop(toDrop)
    timeToEvent = timeToEvent.astype("float")
    survived = survived[boolIndexing]
    annotation.drop(toDrop)
    dlFiles = dlFiles[boolIndexing]

    # Read files and setup data matrix
    counts = []
    allReads = []
    order = []
    for f in dlFiles:
        try:
            fid = f.split(".")[0]
            status = pd.read_csv(paths.countsTCGA + "500centroid/" + fid + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
            counts.append(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values)
            status = status.drop("Unassigned_Unmapped", axis=1)
            allReads.append(status.values.sum())
            order.append(fid)
        except:
            continue
    if len(counts) <= 20:
        print(case, f"not enough samples")
        continue
    if np.logical_not(survived).sum() <= 3:
        print(case, f"not enough survival information")
        continue
    try:
        os.mkdir(paths.outputDir + "rnaseq/Survival/" + case)
    except FileExistsError:
        pass
    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T
    # Normalize and transform counts
    counts = allCounts
    nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=3)
    countsNz = allCounts[:, nzCounts]
    sf = rnaseqFuncs.scranNorm(countsNz)
    # No need to compute full pearson for survival(no need to rescale/compute variance) 
    countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)     
    residuals = countModel.residuals
    normed = countModel.normed
    mappedProbe = np.isin(np.arange(len(consensuses))[nzCounts], multiProg[3].values)
    df = pd.DataFrame()
    df["Dead"] = np.logical_not(survived[np.isin(timeToEvent.index, order)])
    df["TTE"] = timeToEvent.loc[order].values
    df["TTE"] -= df["TTE"].min() - 1
    colnames = [f"X{i}" for i in range(mappedProbe.sum())]
    df[colnames] = residuals[:, mappedProbe]
    df.index = order
    df = df.copy()
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(df)
    for i in range(mappedProbe.sum()):
        try:
            fname = "_".join(multiProg.iloc[i][[0,1,2,3,4]].astype("str")) + "/"
            os.mkdir(paths.outputDir + "rnaseq/Survival/Kaplan150/" + fname)
        except FileExistsError:
            pass
        selectedCol = f"X{i}"
        fml = ro.r(f"Surv(TTE, Dead) ~ {selectedCol}")
        set_seed = ro.r('set.seed')
        set_seed(42)
        mstat = maxstat.maxstat_test(fml, data=r_dataframe, smethod="LogRank", pmethod="condMC", B=100000)
        mp = mstat.rx2('p.value')[0]
        cutoff = mstat.rx2('estimate')[0]
        print(mp, cutoff)
        plt.figure(dpi=500)
        groups = np.where((df[selectedCol] > cutoff).values, "High Expression", "Low Expression")
        kmf = km.fit(df["TTE"], df["Dead"].astype(int), groups)
        km.plot(kmf, title=f"Survival function\n Maximally Selected Logrank test p-value : {mp}")
        plt.ylim(0,1)
        plt.savefig(paths.outputDir + "rnaseq/Survival/Kaplan150/" + fname + case + ".pdf",
                    bbox_inches="tight")
        # plt.show()
        plt.close()
# %%
files = os.listdir(paths.outputDir + "rnaseq/Survival/")
files = [f for f in files if f.startswith("TCGA-")]
counts = pd.Series()
for f in files:
    with open(paths.outputDir + "rnaseq/Survival/" + f + "/prognostic.bed") as fo:
        numProg = len(fo.readlines())
        counts[f] = numProg
# %%
counts.sort_values(ascending=False,inplace=True)
sns.barplot(counts)
# %%
with open(paths.tempDir + "end0505.txt", "w") as f:
    f.write("1")