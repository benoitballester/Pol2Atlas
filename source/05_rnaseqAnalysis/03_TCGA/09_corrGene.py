# %%
import os

import fastcluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import sklearn.metrics as metrics
import umap
from lib import rnaseqFuncs
from lib.utils import matrix_utils, plot_utils
from matplotlib.patches import Patch
from scipy.spatial.distance import dice
from scipy.stats import chi2, rankdata
from settings import params, paths
from statsmodels.stats.multitest import fdrcorrection

folder = paths.outputDir + "rnaseq/TCGA2/corr_expr/"
try:
    os.mkdir(paths.outputDir + "rnaseq/TCGA2/corr_expr/")
except:
    pass
# %%
allAnnots = pd.read_csv(paths.tcgaAnnot, 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
try:
    os.mkdir(paths.outputDir + "rnaseq/TumorVsNormal/")
except FileExistsError:
    pass
perCancerDE = pd.DataFrame()
wAccs = pd.DataFrame()
recalls = pd.DataFrame()
precisions = pd.DataFrame()
studiedConsensusesCase = dict()
cases = allAnnots["project_id"].unique()
# %%
# Load RNAP2 counts
annotation = pd.read_csv(paths.tcgaAnnot, 
                        sep="\t", index_col=0)
annotation.drop_duplicates("Sample ID", inplace=True)
dlFiles = os.listdir(paths.countsTCGA + "500centroid/")
dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
ids = np.array([f.split(".")[0] for f in dlFiles])
inAnnot = np.isin(ids, annotation.index)
ids = ids[inAnnot]
dlFiles = np.array(dlFiles)[inAnnot]
annotation = annotation.loc[ids]
labels = []
for a in annotation["Sample Type"]:
    if a == "Solid Tissue Normal":
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)
# Read files and setup data matrix
counts = []
allReads = []
order = []
for f in dlFiles:
    try:
        fid = f.split(".")[0]
        status = pd.read_csv(paths.countsTCGA + "500centroid/" + fid + ".counts.summary",
                            header=None, index_col=0, sep="\t", skiprows=1).T
        counts.append(pd.read_csv(paths.countsTCGA + "500centroid/" + f, header=None, skiprows=2).values.astype("int32"))
        status = status.drop("Unassigned_Unmapped", axis=1)
        allReads.append(status.values.sum())
        order.append(fid)
    except:
        continue
allReads = np.array(allReads)
allCounts = np.concatenate(counts, axis=1).T
annotation = annotation.loc[order]
# %%
# Read table and annotation, match samples with Pol II
geneTable = pd.read_hdf(paths.tcgaGeneCounts)
geneTableAnnot = pd.read_csv(paths.tcgaAnnotCounts, index_col="Sample ID", sep="\t")
geneTableAnnot = geneTableAnnot[~geneTableAnnot.index.duplicated(keep='first')]
inGeneTable = np.isin(annotation["Sample ID"], geneTableAnnot.index)
used = geneTableAnnot.loc[annotation["Sample ID"][inGeneTable]]["File ID"]
used = used[~used.index.duplicated(keep='first')]
usedTable = geneTable[used].astype("int32").iloc[:-5].T
nzCounts = rnaseqFuncs.filterDetectableGenes(usedTable.values, readMin=1, expMin=3)
usedTable = usedTable.loc[:, nzCounts]
# %%
allCounts = allCounts[inGeneTable]
annotation = annotation[inGeneTable]
# %%
# Compute size factor
sf = rnaseqFuncs.scranNorm(usedTable.values)
sf /= sf.mean()

# %%
# Replace with human readable gene name
ensemblToID = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data/ensembl_toGeneId.tsv", sep="\t", index_col="Gene stable ID")
ensemblToID = ensemblToID[~ensemblToID.index.duplicated(keep='first')]
geneStableID = [id.split(".")[0] for id in usedTable.columns]
valid = np.isin(geneStableID, ensemblToID.index)
usedTable = usedTable.iloc[:, valid]
usedTable.columns = ensemblToID.loc[np.array(geneStableID)[valid]].values.ravel()
# %%
# Normalized count table
normedGeneTable = usedTable/sf[:, None]
normedP2Table = allCounts/sf[:, None]
# %%
from lib import rnaseqFuncs
from joblib.externals.loky import get_reusable_executor
modelGene = rnaseqFuncs.RnaSeqModeler().fit(usedTable.values, sf, residuals="deviance")
modelP2 = rnaseqFuncs.RnaSeqModeler().fit(allCounts, sf, residuals="deviance")
get_reusable_executor().shutdown(wait=False)
# %%
# Compute p-value from query probe-gene pairs
cancerType, eq = pd.factorize(annotation["project_id"], sort=True)
palette, colors = plot_utils.getPalette(cancerType)

# %%
from lib.geneCorrelation import geneCorreler
import pyranges as pr
correler = geneCorreler(paths.GOfile,
                        geneFile=paths.gencode,
                        chrFile=paths.genomeFile, 
                        distal=0, upstream=500000, downstream=500000)
# Load Pol II consensuses
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
# %%
links = correler.fit(consensuses, normedP2Table, normedGeneTable, nPerms=100000)
tab = correler.findCorrelations(linkedGenes=links, alternative="greater")
# %%

# %%
from lib.utils import utils
utils.dump(tab, paths.tempDir + "corrtabTCGA")
utils.dump(correler.randomCorrelations, paths.tempDir + "randomTCGA")
# %%
""" corr_name = utils.load(paths.tempDir + "corrNameTCGA")
correlations = utils.load(paths.tempDir + "correlationsTCGA")
corrP = utils.load(paths.tempDir + "corrPTCGA")
means = utils.load(paths.tempDir + "meansTCGA")
randomCorrelations = utils.load(paths.tempDir + "randomTCGA") """
# %%
tab = pd.DataFrame(tab, columns=["Probe ID", "Gene", "Spearman r", "P-value"])
tab.to_csv(folder + "correlationTab.tsv", sep="\t", index=None)
# %%
# Plot association with end of gene
polII_tail = pd.read_csv(paths.outputDir + "/dist_to_genes/pol2_5000_TES_ext.bed", sep="\t")
polII_tail.index = polII_tail["Name"]
tailP2 = np.isin(tab.iloc[:, 0].astype(int).values, polII_tail["Name"].values)
inter = np.zeros_like(tailP2)
for i in range(len(tailP2)):
    if tailP2[i]:
        gene = tab.iloc[i, 1]
        if gene in polII_tail["gene_name"].loc[int(tab.iloc[i, 0])]:
            inter[i] = True
orderCorr = np.argsort(tab["Spearman r"].astype(float).values)
plt.figure(dpi=500)
plt.ylabel("Spearman r")
plt.xlabel("Rank")
plt.scatter(np.arange(len(orderCorr)), tab["Spearman r"].astype(float).values[orderCorr], s=2.0, c=inter[orderCorr]*1.0, linewidths=0)
plt.savefig(folder + "allCorrFlagTailSame.png")
plt.figure(dpi=500)
plt.ylabel("Spearman r")
plt.xlabel("Rank")
plt.scatter(np.arange(len(orderCorr)), tab["Spearman r"].astype(float).values[orderCorr], s=2.0, c=tailP2[orderCorr]*1.0, linewidths=0)
plt.savefig(folder + "allCorrFlagTail.png")
# %%
# End of gene @ 5% FDR greater
sig, fdr = fdrcorrection(tab["P-value"].astype("float").values)
sigTab = tab[sig]
sigTab["FDR"] = fdr[sig]
sigTab.to_csv(folder + "correlationTab_sigPosCor.tsv", sep="\t", index=None)
tailP2Sig = np.isin(sigTab.iloc[:, 0].astype(int).values, polII_tail["Name"].values)
interSig = np.zeros_like(tailP2Sig)
for i in range(len(tailP2Sig)):
    if tailP2Sig[i]:
        gene = sigTab.iloc[i, 1]
        if gene in polII_tail["gene_name"].loc[int(sigTab.iloc[i, 0])]:
            interSig[i] = True
print(f"Overrepresentation of Pol II tail @ 5000 same gene: {(interSig.mean())/(inter.mean())}")
print(f"Overrepresentation of Pol II tail @ 5000 : {(tailP2Sig.mean())/(tailP2.mean())}")
print(f"Fraction of Pol II tails @ 5000 same gene : {(interSig.mean())}")
print(f"Fraction of Pol II tails @ 5000 : {(tailP2Sig.mean())}")
print(f"Number of gene-probe pairs 'tail same gene' : {interSig.sum()}")
print(f"Number of gene-probe pairs 'tail' : {tailP2Sig.sum()}")
orderCorr = np.argsort(sigTab["Spearman r"].astype(float).values)
plt.figure(dpi=500)
plt.ylabel("Spearman r")
plt.xlabel("Rank")
plt.scatter(np.arange(len(orderCorr)), sigTab["Spearman r"].astype(float).values[orderCorr], s=2.0, c=interSig[orderCorr]*1.0, linewidths=0)
plt.savefig(folder + "sigCorrFlagTailSame.png")
plt.figure(dpi=500)
plt.ylabel("Spearman r")
plt.xlabel("Rank")
plt.scatter(np.arange(len(orderCorr)), sigTab["Spearman r"].astype(float).values[orderCorr], s=2.0, c=tailP2Sig[orderCorr]*1.0, linewidths=0)
plt.savefig(folder + "sigCorrFlagTail.png")
# %%
# End of gene @ 5% FDR greater
sigTwo, fdr = fdrcorrection(np.minimum(np.array(tab["P-value"].astype("float").values),1-np.array(tab["P-value"].astype("float").values))*2)
sigTab = tab[sigTwo]
sigTab["P-value"] = (np.minimum(np.array(tab["P-value"].astype("float").values),1-np.array(tab["P-value"].astype("float").values))*2)[sigTwo]
sigTab["FDR"] = fdr[sigTwo]
sigTab.to_csv(folder + "correlationTab_sigTwosided.tsv", sep="\t", index=None)
# %%
from sklearn.preprocessing import StandardScaler
corrPol = []
corrGene = []
i = 0
valid_corr = np.array(tab.iloc[:, [0,1]])[sig]
""" corrPol = modelP2.residuals[:, valid_corr[:, 0].astype(int)]
reorder = pd.Series(np.arange(len(normedGeneTable.columns)), 
                    index=normedGeneTable.columns).loc[valid_corr[:, 1]].values
corrGene = modelGene.residuals[:, reorder] """
corrPol = normedP2Table[:, valid_corr[:, 0].astype(int)]
corrGene = normedGeneTable[valid_corr[:, 1]].values
corrPol = StandardScaler().fit_transform(np.log1p(corrPol))
corrGene = StandardScaler().fit_transform(np.log1p(corrGene))
# %%
decomp = rnaseqFuncs.permutationPA_PCA(corrPol, perm=3, max_rank=1000)
rowOrder, rowLink = matrix_utils.threeStagesHClinkage(decomp, metric="correlation")
colOrder, colLink = matrix_utils.threeStagesHClinkage(corrPol.T, metric="euclidean")
# %%
from skimage.transform import rescale, resize, downscale_local_mean
resized1 = resize(corrGene[rowOrder][:, colOrder], (2000, 3000), anti_aliasing=True, order=1)
resized2 = resize(corrPol[rowOrder][:, colOrder], (2000, 3000), anti_aliasing=True, order=1)
# %%
plt.figure()
plt.imshow(np.transpose(colors[rowOrder].reshape(-1,1,3), [1,0,2]), interpolation="lanczos")
plt.gca().set_aspect(1000)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig(folder + "labels.pdf")
# %%
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,2, dpi=500) 
plot1 = axarr[1].imshow(resized1.T, cmap=sns.color_palette("viridis", as_cmap=True), vmin=-2, vmax=2)
axarr[1].title.set_text("Genes")
axarr[1].set_aspect(1.5)
cbar1 = plt.colorbar(plot1,ax=axarr[1])
cbar1.set_label("Z-score")
axarr[1].set(xlabel="Samples")
plot2 = axarr[0].imshow(resized2.T, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-2, vmax=2)
cbar2 = plt.colorbar(plot2,ax=axarr[0])
cbar2.set_label("Z-score")
axarr[0].title.set_text("Pol2 probes")
axarr[0].set_aspect(1.5)
axarr[0].set(xlabel="Samples")
for ax in axarr:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel(f"{corrGene.shape[1]} Pol II probes-gene links")
plt.savefig(folder + "heatmaps.pdf")
# %%
from scipy.cluster import hierarchy
plt.figure(dpi=500)
hierarchy.dendrogram(colLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.show()
plt.savefig(folder + "linksDendrogram.pdf")
plt.close()
# %%
from scipy.cluster import hierarchy
plt.figure(dpi=500)
hierarchy.dendrogram(rowLink, p=10, truncate_mode="level", color_threshold=-1)
plt.axis('off')
plt.show()
plt.savefig(folder + "SampleDendrogram.pdf")
plt.close()
# %%
noTail = np.logical_not(tailP2Sig)
colOrder, colLink = matrix_utils.threeStagesHClinkage(corrPol.T[noTail], metric="euclidean")
# %%
from skimage.transform import rescale, resize, downscale_local_mean
resized1 = resize(corrGene[rowOrder][:,noTail][:, colOrder], (2000, 3000), anti_aliasing=True, order=1)
resized2 = resize(corrPol[rowOrder][:,noTail][:, colOrder], (2000, 3000), anti_aliasing=True, order=1)
# %%
plt.figure()
plt.imshow(np.transpose(colors[rowOrder].reshape(-1,1,3), [1,0,2]), interpolation="lanczos")
plt.gca().set_aspect(1000)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig(folder + "labels.pdf")
# %%
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,2, dpi=500) 
plot1 = axarr[1].imshow(resized1.T, cmap=sns.color_palette("viridis", as_cmap=True), vmin=-2, vmax=2)
axarr[1].title.set_text("Genes")
axarr[1].set_aspect(1.5)
cbar1 = plt.colorbar(plot1,ax=axarr[1])
cbar1.set_label("Z-score")
axarr[1].set(xlabel="Samples")
plot2 = axarr[0].imshow(resized2.T, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-2, vmax=2)
cbar2 = plt.colorbar(plot2,ax=axarr[0])
cbar2.set_label("Z-score")
axarr[0].title.set_text("Pol2 probes")
axarr[0].set_aspect(1.5)
axarr[0].set(xlabel="Samples")
for ax in axarr:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel(f"{colOrder.shape[0]} Pol II probes-gene links")
plt.savefig(folder + "heatmaps_noTail.pdf")
# %%
globallyDE = pd.read_csv(paths.outputDir + "rnaseq/TumorVsNormal/globally_DE.bed", sep="\t", header=None)
globallyDE.index= globallyDE[3]
# %%
from scipy.stats import spearmanr
gene = "TGFBR2"
p2 = 108662
exprGene = normedGeneTable[gene].values
exprP2 = normedP2Table[:, p2]
r, p = spearmanr(np.log10(exprGene+1), np.log10(exprP2+1))
p = (r < correler.randomCorrelations[:, 0]).mean()
print(r, p)
plt.figure(dpi=300, figsize=(3,3))
plt.scatter(np.log10(exprGene+1), np.log10(exprP2+1), s=1.0, linewidths=0, c=colors)
plt.xlabel(f"{gene} expression")
plt.ylabel(f"Probe #{p2} expression")
plt.title(f"Spearman r : {int(r*1000)/1000.0}\n Link permutation p-value : {int(p*10000)/10000.0}")
plt.savefig(folder + "corr_TGFBR2_108662.pdf")
# %%
sigTab = tab[sig][noTail]
listDECorr = globallyDE.loc[globallyDE.index.intersection(sigTab["Probe ID"].astype(int).values)]
# %%
with open(paths.tempDir + "end0509.txt", "w") as f:
    f.write("1")
