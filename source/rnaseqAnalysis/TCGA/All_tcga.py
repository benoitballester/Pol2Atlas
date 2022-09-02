# %%
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
from settings import params, paths
from lib import rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2
from scipy.stats import chi2
import seaborn as sns
import umap
from statsmodels.stats.multitest import fdrcorrection
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
scran = importr("scran")
# %%
annotation = pd.read_csv(paths.tcgaData + "/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
dlFiles = os.listdir(paths.countsTCGA + "BG/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
counts = []
countsBG = []
allReads = []
order = []
for f in np.array(dlFiles):
    try:
        id = f.split(".")[0]
        # countsBG.append(pd.read_csv(paths.countsTCGA + "BG/" + f, header=None, skiprows=2).values)
        status = pd.read_csv(paths.countsTCGA + "500centroid/" + id + ".counts.summary",
                             header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(id)
    except:
        continue
allReads = np.array(allReads)
counts = np.concatenate(counts, axis=1).T
# bgCounts = np.concatenate(countsBG, axis=1).T
# %%
# Keep tumoral samples
kept = np.isin(order, annotation.index)
counts = counts[kept]
# bgCounts = bgCounts[:, kept]
allReads = allReads[kept]
annotation = annotation.loc[np.array(order)[kept]]
tumor = np.logical_not(annotation["Sample Type"] == "Solid Tissue Normal")
# %%
# Remove undected Pol II probes
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
counts = counts[:, nzCounts]
# %%
# Normalize
sf = rnaseqFuncs.scranNorm(counts)
# %%
try:
    os.mkdir(paths.outputDir + "rnaseq/TCGA/")
except FileExistsError:
    pass
# %%
# Feature selection 
countModel = rnaseqFuncs.RnaSeqModeler().fit(counts, sf)
hv = countModel.hv
# %%
# Identify DE per condition
try:
    os.mkdir(paths.outputDir + "rnaseq/TCGA/DE/")
except FileExistsError:
    pass
# %%
from lib.pyGREATglm import pyGREAT as pyGREATglm
enricherglm = pyGREATglm(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd"]
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
except FileExistsError:
    pass
# %%
# 1 vs All DE analysis (markers)
from lib.utils.reusableUtest import mannWhitneyAsymp
tester = mannWhitneyAsymp(countModel.residuals)
pctThreshold = 0.1
lfcMin = 0.25
orig = annotation["Project ID"]
state = np.array(["Normal"]*len(orig))
state[tumor] = "Tumor"
annotConv = pd.read_csv(paths.tcgaToMainAnnot, sep="\t", index_col=0)
orig = annotConv.loc[orig]["Origin"]
label = orig.str.cat(state,sep="_")

for i in label.unique():
    print(i)
    labels = (label == i).astype(int)
    res2 = tester.test(labels, "less")
    sig = fdrcorrection(res2[1])[0]
    minpct = np.mean(counts[label == i] > 0.5, axis=0) > max(0.1, 1.5/labels.sum())
    fc = np.mean(counts[label == i], axis=0) / (1e-9+np.mean(counts[label != i], axis=0))
    lfc = np.log2(fc) > lfcMin
    sig = sig & lfc & minpct
    print(sig.sum())
    res = pd.DataFrame(res2[::-1], columns=consensuses.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = sig.astype(int)
    res.to_csv(paths.outputDir + f"rnaseq/TCGA/DE/res_{i}.csv")
    test = consensuses[nzCounts][sig]
    test.to_csv(paths.outputDir + f"rnaseq/TCGA/DE/bed_{i}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/TCGA/DE/great_{i}.pdf")

# %%
# Compute PCA
feat = countModel.residuals[:, hv]
decomp, model = rnaseqFuncs.permutationPA_PCA(feat, max_rank=2000, returnModel=True)
# %%
# UMAP
import umap
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=0, low_memory=False, 
                      metric="correlation").fit_transform(decomp)
# %%
# Plot UMAP
import plotly.express as px
import plotly.graph_objects as go
df = pd.DataFrame(embedding, columns=["x","y"])
df["Sample Type"] = annotation["Sample Type"].values
df["Project"] = annotation["Project ID"].values
df["Normal"] = 1.0*(annotation["Sample Type"].values == "Solid Tissue Normal")
df["Size"] = 3*np.sqrt(len(df)/7500) * (1+df["Normal"])
df["Hover"] = df["Project"] + "\n" + df["Sample Type"]
project_id = annotation["project_id"]
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
palette, colors = getPalette(cancerType)
df["Color"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
markers = go.scattergl.Marker(color=df["Color"], size=df["Size"], 
                            line=dict(width=df["Normal"], color="rgb(0,0,0)"))
dat = go.Scattergl(x=embedding[:,0],y=embedding[:,1], mode="markers",
                   marker=markers, hovertext=df["Hover"])
layout = dict(height=800, width=1200)
fig = go.Figure(dat, layout=layout)
fig.show()
fig.write_image(paths.outputDir + "rnaseq/TCGA/umap_samples.pdf")
fig.write_html(paths.outputDir + "rnaseq/TCGA/umap_samples.pdf" + ".html")
# %%
import plotly.express as px
#%%
# Predictive model
import catboost
from sklearn.model_selection import train_test_split, StratifiedKFold
project_id = annotation["project_id"]
# cancerType, eq = pd.factorize(tcgaProjects["Origin"].loc[project_id])
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
pred = np.zeros(len(cancerType), dtype=int)
for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(decomp, cancerType):
    # Fit power transform on train data only
    x_train = decomp[train]
    # Fit classifier on scaled train data
    model = catboost.CatBoostClassifier(class_weights=len(cancerType) / (len(np.unique(cancerType)) * np.bincount(cancerType)), random_state=42)
    model.fit(x_train, cancerType[train], silent=True)
    # Scale and predict on test data
    x_test = decomp[test]
    pred[test] = model.predict(x_test).ravel()
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
acc = balanced_accuracy_score(cancerType, pred)
# %%
# Legend of UMAP + balanced accuracy
from lib.utils.plot_utils import plotUmap, getPalette
from matplotlib.patches import Patch

plt.figure(dpi=500)
palette, colors = getPalette(cancerType)
# allReadsScaled = (allReads - allReads.min()) / (allReads.max()-allReads.min())
# col = sns.color_palette("rocket_r", as_cmap=True)(allReadsScaled)
scale = min(10.0,100/np.sqrt(len(embedding)))
plt.scatter(embedding[:, 0], embedding[:, 1], s=scale*(3.0-2.0*tumor),
            linewidths=(1.0-tumor)*0.5, edgecolors=(0.1,0.1,0.1),c=colors)
xScale = plt.xlim()[1] - plt.xlim()[0]
yScale = plt.ylim()[1] - plt.ylim()[0]
# plt.gca().set_aspect(xScale/yScale)
plt.axis('off')
plt.savefig(paths.outputDir + "rnaseq/TCGA/umap_all_tumors_plus_normal.png")
plt.show()
plt.figure(dpi=500)
plt.axis('off')
patches = []
for i in np.unique(cancerType):
    legend = Patch(color=palette[i], label=eq[i][5:])
    patches.append(legend)
plt.legend(handles=patches)
plt.title(f"Balanced accuracy on {len(np.unique(cancerType))} cancers : {acc}")
plt.savefig(paths.outputDir + "rnaseq/TCGA/umap_all_tumors_lgd.png", bbox_inches="tight")
plt.show()

# %%
'''
clustsPol2 = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt",dtype=int)[nzCounts]
nClusts = np.max(clustsPol2)+1
nAnnots = len(eq)
zScores = np.zeros((nClusts, nAnnots))
avg1Read = np.mean(countModel.residuals, axis=0) > 1
filteredMat = np.log(1+countModel.residuals)[:, avg1Read]
for i in range(nAnnots):
    hasAnnot = cancerType == i
    sd = np.std(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    expected = np.mean(np.percentile(filteredMat[hasAnnot], 95, axis=0))
    for j in range(nClusts):
        inClust = clustsPol2[avg1Read] == j
        notInClust = np.logical_not(clustsPol2 == j)
        observed = np.mean(np.percentile(filteredMat[hasAnnot][:, inClust], 95, axis=0))
        zScores[j, i] = (observed-expected)/sd
import matplotlib as mpl
rowOrder, colOrder = matrix_utils.HcOrder(zScores)
rowOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
zClip = np.clip(zScores,0.0,10.0)
zNorm = np.clip(zClip / np.percentile(zClip, 95),0.0,1.0)
plt.figure(dpi=300)
sns.heatmap(zNorm[rowOrder].T[colOrder], cmap="vlag", linewidths=0.1, linecolor='black', cbar=False)
plt.gca().set_aspect(2.0)
plt.yticks(np.arange(len(eq))+0.5, eq[colOrder], rotation=0)
plt.xticks([],[])
plt.xlabel(f"{len(zNorm)} Pol II clusters")
plt.savefig(paths.outputDir + "rnaseq/TCGA/signalPerClustPerAnnot.pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 1), dpi=300)
norm = mpl.colors.Normalize(vmin=0, vmax=np.percentile(zClip, 95))
cb = mpl.colorbar.ColorbarBase(plt.gca(), sns.color_palette("vlag", as_cmap=True), norm, orientation='horizontal')
cb.set_label("95th percentile Z-score")
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/TCGA/signalPerClustPerAnnot_colorbar.pdf")
plt.show()
'''
# %%
# HC 
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T, "correlation")
# Plot Heatmap
vals = countModel.normed
plot_utils.plotHC(vals.T, eq[cancerType], vals.T,  
                    rowOrder=rowOrder, colOrder=colOrder, hq=True)
plt.savefig(paths.outputDir + "rnaseq/TCGA/HM_all.pdf", bbox_inches="tight")
# %%
# HC (hv probes only)
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, "correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(countModel.residuals.T[hv], "correlation")
# Plot Heatmap
vals = countModel.normed
plot_utils.plotHC(vals.T[hv], eq[cancerType], vals.T[hv],  
                    rowOrder=rowOrder, colOrder=colOrder, hq=True)
plt.savefig(paths.outputDir + "rnaseq/TCGA/HM_hv.pdf", bbox_inches="tight")

# %%
