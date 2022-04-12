# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from settings import params, paths
from lib import normRNAseq, rnaseqFuncs
from lib.utils import plot_utils, matrix_utils
from matplotlib.patches import Patch
from scipy.stats import rankdata, chi2, mannwhitneyu, ttest_ind
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
import lifelines as ll
from joblib import Parallel, delayed
import warnings
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from matplotlib.ticker import FormatStrFormatter
numpy2ri.activate()
scran = importr("scran")
np.random.seed(42)
# %%
from lib.pyGREATglm import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
import pyranges as pr
from lib.utils import overlap_utils
remap_catalog = pr.read_bed(paths.remapFile)
# %%
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
try:
    os.mkdir(paths.outputDir + "rnaseq/Survival2/")
except FileExistsError:
    pass
progPerCancer = pd.DataFrame()
statsCase = dict()
studiedConsensusesCase = dict()
nzPerCancer = dict()
cases = allAnnots["project_id"].unique()
# %%
for case in cases:
    print(case)
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
            status = pd.read_csv(paths.countDirectory + "500centroid/" + fid + ".counts.summary",
                                header=None, index_col=0, sep="\t", skiprows=1).T
            counts.append(pd.read_csv(paths.countDirectory + "500centroid/" + f, header=None, skiprows=2).values)
            status = status.drop("Unassigned_Unmapped", axis=1)
            allReads.append(status.values.sum())
            order.append(fid)
        except:
            continue
    if len(counts) <= 20:
        print(case, f"not enough samples ({allCounts.shape})")
        continue
    if np.logical_not(survived).sum() < 3:
        print(case, f"not enough survival information")
        continue
    try:
        os.mkdir(paths.outputDir + "rnaseq/Survival2/" + case)
    except FileExistsError:
        pass
    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T
    # Normalize and transform counts
    counts = allCounts
    
    nzCounts = rnaseqFuncs.filterDetectableGenes(allCounts, readMin=5, expMin=2)
    countsNz = allCounts[:, nzCounts]
    sf = scran.calculateSumFactors(countsNz.T)
    countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)         
    anscombe = countModel.anscombeResiduals
    countsNorm = countModel.normed
    df = pd.DataFrame()
    df["Dead"] = np.logical_not(survived[np.isin(timeToEvent.index, order)])
    df["TTE"] = timeToEvent.loc[order].values
    df.index = order
    df = df.copy()

    stats = []
    cutoffs = []
    notDropped = []
    print(f"Cox regression on {anscombe.shape[1]} peaks")
    # Compute univariate cox proportionnal hazards p value 
    def computeCoxReg(expr, survDF, i):
        warnings.filterwarnings("ignore")
        try:
            dfI = survDF.copy()
            dfI[i] = expr[:, i]
            cph = ll.CoxPHFitter()
            cph.fit(dfI[[i, "TTE", "Dead"]], duration_col="TTE", event_col="Dead", robust=True)
            stats = cph.summary
            if np.isnan(stats["p"].values[0]):
                raise ValueError('Pval is NaN')
            return cph.summary
        except:
            # If the regression failed to converge assume HR=1.0 and p = 1.0
            dummyDF = pd.DataFrame(data=[[0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0]], 
                                    columns=['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)'], 
                                    index=[i])
            dummyDF.index.name = "covariate"
            return dummyDF
    # Parallelize Cox regression across available cores using joblib
    batchSize = min(512, int(anscombe.shape[1]/len(os.sched_getaffinity(0))))

    with Parallel(n_jobs=32, verbose=1, batch_size=batchSize) as pool:
        stats = pool(delayed(computeCoxReg)(anscombe, df, i) for i in range(anscombe.shape[1]))
    stats = pd.concat(stats)
    stats.index = nzCounts.nonzero()[0]
    pvals = np.array(stats["p"])
    statsCase[case] = stats
    qvals = fdrcorrection(pvals)[1]
    consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
    progConsensuses = consensuses[nzCounts][qvals < 0.05]
    progConsensuses[4] = stats["exp(coef)"].loc[progConsensuses[3]]
    progConsensuses.to_csv(paths.outputDir + "rnaseq/Survival2/" + case + "/prognostic.bed", sep="\t", header=None, index=None)
    stats.to_csv(paths.outputDir + "rnaseq/Survival2/" + case + "/stats.csv", sep="\t")
    enrichedGREAT = enricher.findEnriched(progConsensuses, consensuses[nzCounts])
    enrichedGREAT.to_csv(paths.outputDir + "rnaseq/Survival2/" + case + "/GREATenriched.csv", sep="\t")
    enricher.plotEnrichs(enrichedGREAT, savePath=paths.outputDir + "rnaseq/Survival2/" + case + "/GREATenriched.pdf")
    studiedConsensusesCase[case] = nzCounts.nonzero()[0]
    progPerCancer[case] = np.zeros(allCounts.shape[1])
    progPerCancer[case][nzCounts] = np.where(qvals > 0.05, 0.0, np.sign(stats["coef"].ravel()))
# %%
# Plot # of DE Pol II Cancer vs Normal
DEperCancer = pd.DataFrame(np.sum(np.abs(progPerCancer), axis=0)).T
plt.figure(figsize=(6,4), dpi=500)
order = np.argsort(DEperCancer).values[0]
plt.barh(np.arange(len(DEperCancer.T)),DEperCancer.values[0][order])
plt.yticks(np.arange(len(DEperCancer.columns)), DEperCancer.columns[order], fontsize=8)
plt.xlabel("# of Prognostic Pol II Cancer vs Normal")
plt.ylabel("Cancer type")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(paths.outputDir + "rnaseq/Survival2/prog_countPerCancer.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(6,4), dpi=500)
order = np.argsort(DEperCancer).values[0]
plt.xscale("symlog")
plt.barh(np.arange(len(DEperCancer.T)),DEperCancer.values[0][order])
plt.yticks(np.arange(len(DEperCancer.columns)), DEperCancer.columns[order], fontsize=8)
plt.xlabel("# of Prognostic Pol II Cancer vs Normal")
plt.ylabel("Cancer type")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.savefig(paths.outputDir + "rnaseq/Survival2/prog_countPerCancer_log.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
mat = np.abs(progPerCancer).astype("int32")
consensusDECount = np.sum(mat, axis=1).astype("int32")
# Compute null distribution of DE Pol II
countsRnd = np.zeros(mat.shape[1])
nPerm = 100
for i in range(nPerm):
    shuffledDF = np.zeros_like(mat)
    for j, cancer in enumerate(mat.columns):
        # Permute only on detected Pol II
        shuffledDF[studiedConsensusesCase[cancer], j] = np.random.permutation(mat.iloc[studiedConsensusesCase[cancer], j])
    s = np.sum(shuffledDF, axis=1)
    counts = np.bincount(s)/nPerm
    countsRnd[:counts.shape[0]] += counts
countsObs = np.bincount(np.sum(mat, axis=1))
for threshold in range(len(countsObs)):
    randomSum = np.sum(countsRnd[threshold:])
    fpr = randomSum / (np.sum(countsObs[threshold:])+randomSum)
    print(threshold, fpr)
    if fpr < 0.05:
        break
studied = np.sum(mat, axis=1) >= threshold
plt.figure(figsize=(6,4), dpi=300)
plt.hist(consensusDECount, np.arange(consensusDECount.max()+1))
plt.hist(s, np.arange(consensusDECount.max()+1), alpha=0.5)
plt.xlabel(f"prognostic in x cancers")
plt.ylabel("# of Pol II")
plt.legend(["Observed", "Expected"])
plt.vlines(threshold + 0.0, plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
plt.text(threshold + 0.25, plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, "FPR < 5%", color="red")
plt.xticks(np.arange(0,consensusDECount.max()+1)+0.5, np.arange(0,consensusDECount.max()+1))
plt.savefig(paths.outputDir + f"rnaseq/Survival2/multiple_prognostic.pdf", bbox_inches="tight")
plt.show()
globallyDEs = consensuses[studied]
globallyDEs[4] = np.sum(mat, axis=1)
globallyDEs.to_csv(paths.outputDir + f"rnaseq/Survival2/globally_prognostic.bed", sep="\t", header=None, index=None)
enrichs = enricher.findEnriched(consensuses[studied], consensuses)
enricher.plotEnrichs(enrichs, savePath=paths.outputDir + "rnaseq/Survival2/globally_prognostic_GREAT.pdf")
enrichs.to_csv(paths.outputDir + f"rnaseq/Survival2/globally_prognostic_GREAT.csv", sep="\t")
# %%
# Forest Plots
try:
    os.mkdir(paths.outputDir + "rnaseq/Survival2/global_forestPlots/")
except FileExistsError:
    pass
globallyDEs = pd.read_csv(paths.outputDir + f"rnaseq/Survival2/globally_prognostic.bed", sep="\t", header=None)
for i, cons in globallyDEs.iterrows():
    df = pd.DataFrame()
    for c in statsCase:
        try:
            stats = statsCase[c]
        except KeyError:
            continue
        queryCons = cons[3]
        try:
            stats = stats.loc[queryCons]
        except KeyError:
            continue
        stats.name = c
        df = pd.concat([df,pd.DataFrame(stats).T])
    coord = "_".join(cons[[0,1,2,3,4]].astype("str"))
    plt.figure()
    plot_utils.forestPlot(df)
    plt.savefig(paths.outputDir + f"rnaseq/Survival2/global_forestPlots/{coord}.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
# %%
globallyDEs = pd.read_csv(paths.outputDir + f"rnaseq/Survival2/globally_prognostic.bed", sep="\t", header=None)
ids = globallyDEs[3]
coefPerCancer = []
qvalsPerCancer = []
obsCases = []
for i, c in enumerate(cases):
    try:
        stats = pd.read_csv(paths.outputDir + f"rnaseq/Survival2/{c}/stats.csv", sep="\t", index_col=0)
    except FileNotFoundError:
        continue
    idsInCase = np.isin(ids, stats.index)
    progStats = np.zeros(len(globallyDEs))
    progStats[idsInCase] = stats.loc[ids[idsInCase]]["coef"]
    coefPerCancer.append(progStats)
    progP = np.zeros(len(globallyDEs))
    corrP = fdrcorrection(stats["p"])[1]
    progP[idsInCase] = corrP[np.isin(stats.index, ids[idsInCase])]
    qvalsPerCancer.append(progP)
    obsCases.append(c)
coefPerCancer = np.array(coefPerCancer).T
qvalsPerCancer = np.array(qvalsPerCancer).T
# %%
coefPerCancer = pd.DataFrame(coefPerCancer.T, index=obsCases)
sig = pd.DataFrame(np.where(qvalsPerCancer < 0.05, "*", " ").T, index=obsCases)
# %%
plt.figure(dpi=500)
orderCol = np.argsort(coefPerCancer.mean(axis=0))[::-1]
orderRow = np.argsort(np.abs(coefPerCancer).mean(axis=1))[::-1]
sns.heatmap(coefPerCancer[orderCol].iloc[orderRow]/np.log(2), vmin=-1.0, vmax=1.0, cmap="vlag", 
            linecolor="k", linewidths=0.25, cbar_kws={'label': 'log2(HR)'},
            annot=sig[orderCol].iloc[orderRow].values, fmt="", annot_kws={"size": 6})
plt.xticks([])
plt.yticks(np.arange(len(obsCases))+0.5, np.array(obsCases)[orderRow], fontsize=8)
plt.xlabel(f"{len(globallyDEs)} \"Globally prognostic\" Pol II probes")
plt.savefig(paths.outputDir + "rnaseq/Survival2/globally_prognostic_heatmap.pdf", bbox_inches="tight")
# %%
