# %%
from msilib.schema import File
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from lib import rnaseqFuncs
from lib.pyGREATglm import pyGREAT
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.cluster import hierarchy
from settings import paths
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
try:
    os.mkdir(paths.outputDir + "rnaseq/DE_per_dataset/")
except FileExistsError:
    pass
# %%
# Load GTEX counts
countDir = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/gtex_counts/"
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/GTex/tsvs/sample.tsv", 
                        sep="\t", index_col="specimen_id")
colors = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/GTex/colors.txt", 
                        sep="\t", index_col="tissue_site_detail")
dlFiles = os.listdir(countDir + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
allStatus = []
for f in dlFiles:
        id = ".".join(f.split(".")[:-3])
        # countsBG.append(pd.read_csv(paths.countDirectory + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(countDir + "500centroid/" + id + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(countDir + "500centroid/" + f, header=None, skiprows=2).values)
        allStatus.append(status)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(f.split(".")[0])
    
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
ann, eq = pd.factorize(annotation.loc[order]["tissue_type"])
# %%
# Deseq2 DE calculations
nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=1, expMin=2)
countsNz = allCounts[:, nzCounts]
# Scran normalization
sf = rnaseqFuncs.scranNorm(countsNz)
# %%
for i in np.unique(ann):
    print(eq[i])
    labels = (ann == i).astype(int)
    DE_res = rnaseqFuncs.deseqDE(countsNz, sf, labels, order, parallel=True)
    break
# %%
