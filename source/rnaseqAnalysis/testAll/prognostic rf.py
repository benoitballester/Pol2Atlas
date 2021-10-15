# %%
import numpy as np
import pandas as pd
from settings import params, paths
from scipy.io import mmread
import os
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from kneed import KneeLocator
from settings import params, paths
from lib import normRNAseq
from scipy.special import expit
from sklearn.preprocessing import power_transform, PowerTransformer
# %%
case = "TCGA-BRCA"
# %%
# Select only relevant files and annotations
annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
annotation = annotation[annotation["project_id"] == case]
annotation = annotation[np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")]
dlFiles = os.listdir(paths.countDirectory + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]
# %%
# Read survival information
survived = (annotation["vital_status"] == "Alive").values
timeToEvent = annotation["days_to_last_follow_up"].where(survived, annotation["days_to_death"])
timeToEvent = timeToEvent.astype("float")
# %%
# Read files and setup data matrix
counts = []
allReads = []
order = []
for f in dlFiles:
    try:
        fid = f.split(".")[0]
        status = pd.read_csv(paths.countDirectory + "500centroid/" + fid + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(fid)
    except:
        continue
allReads = np.array(allReads)
# %%
allCounts = np.concatenate(counts, axis=1)
# %%
# %%
# Remove low counts + scran deconvolution normalization
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")


countsNorm = allCounts.T / np.mean(allCounts, axis=0)[:, None]
countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
nzPos = np.mean(countsNorm, axis=0) > 1
countsNorm = countsNorm[:, nzPos]
# %%
# Select highly variables regions
dec = scran.modelGeneVar(np.log2(1+countsNorm.T))
mean = np.array(dec.slots["listData"].rx("mean")).ravel()
var = np.array(dec.slots["listData"].rx("total")).ravel()
pval = np.array(dec.slots["listData"].rx["p.value"]).ravel()
fdr = np.array(dec.slots["listData"].rx["FDR"]).ravel()
top = pval < 0.05
c = np.zeros((len(pval), 3)) + np.array([0.0,0.0,1.0])
c[top] = [1.0,0.0,0.0]
plt.figure(dpi=500)
plt.scatter(mean, var, c=c, s=0.5, linewidths=0.0)
plt.show()
# %%
# Yeo-johnson transform and scale to unit variance
countsScaled = countsNorm[:, top]
# %%
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()
maxstat = importr("maxstat")
survival = importr("survival")
df = pd.DataFrame()
df[np.arange(countsScaled.shape[1])] = countsScaled
df["Survived"] = survived[np.isin(timeToEvent.index, order)]
df["TTE"] = timeToEvent.loc[order].values
df.index = order
df = df.copy()
r_dataframe = ro.conversion.py2rpy(df)

# %%
from sksurv.ensemble import RandomSurvivalForest
model = RandomSurvivalForest(n_jobs=-1)
data = np.zeros(len(df), np.dtype({'names':('Cens', 'TTE'), 'formats':('bool', 'int')}))
data["Cens"] = df["Survived"]
data["TTE"] = df["TTE"]
model.fit(countsScaled, data)
# %%
important = perm.feature_importances_ > 2*perm.feature_importances_std_
# %%
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[nzPos][top]
topLocs = consensuses[important]
topLocs.to_csv(paths.tempDir + "topLocsBRCA_prog.csv", sep="\t", header=None, index=None)
topLocs
# %%
import kaplanmeier as km
order = np.argsort(qvals)
for i in order[:10]:
    print(f"Top {i} prognostic")
    plt.figure(dpi=300)
    transfo = np.log2(1+countsScaled[:, notDropped][:, i])
    plt.hist(transfo,50, density=True)
    plt.vlines(np.log2(1+cutoffs[i]), plt.ylim()[0],plt.ylim()[1], color="red")
    plt.vlines(np.log2(1+np.percentile(countsScaled[:, notDropped][:, i],90)), 
                plt.ylim()[0],plt.ylim()[1], color="green")
    plt.vlines(np.log2(1+np.percentile(countsScaled[:, notDropped][:, i],10)), 
                plt.ylim()[0],plt.ylim()[1], color="green")
    plt.xlabel("log2(1 + scran counts)")
    plt.ylabel("Density")
    plt.show()
    groups = countsScaled[:, notDropped][:, i] > cutoffs[i]
    kmf = km.fit(df["TTE"], 
                            df["Survived"], groups)
    km.plot(kmf)
    plt.show()
# %%
