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
from sklearn.preprocessing import power_transform, PowerTransformer, QuantileTransformer

# %%
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
DE_upreg = pd.DataFrame()
DE_downreg = pd.DataFrame()
wAccs = pd.DataFrame()
for case in allAnnots["project_id"].unique():
    print(case)
    # Select only relevant files and annotations
    annotation = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                            sep="\t", index_col=0)
    annotation = annotation[annotation["project_id"] == case]
    dlFiles = os.listdir(paths.countDirectory + "500centroid/")
    dlFiles = [f for f in dlFiles if f.endswith(".txt.gz")]
    ids = np.array([f.split(".")[0] for f in dlFiles])
    inAnnot = np.isin(ids, annotation.index)
    ids = ids[inAnnot]
    dlFiles = np.array(dlFiles)[inAnnot]
    annotation = annotation.loc[ids]

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
    
    labels = []
    for a in annotation["Sample Type"].loc[order]:
        if a == "Solid Tissue Normal":
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)
    if len(labels) < 10 or np.any(np.bincount(labels) < 10):
        print(case, "not enough samples")
        continue

    allReads = np.array(allReads)
    allCounts = np.concatenate(counts, axis=1).T

    # Normalize
    scale = normRNAseq.deseqNorm(allCounts)
    countsNorm = allCounts / scale

    # Predictive model
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    import catboost
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import mannwhitneyu, ttest_ind, ranksums
    from statsmodels.stats.multitest import fdrcorrection
    from sklearn.decomposition import PCA
    labels = []
    for a in annotation["Sample Type"].loc[order]:
        if a == "Solid Tissue Normal":
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)

    predictions = np.zeros(len(labels), dtype=int)
    '''
    for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(countsNorm, labels):
        pvals = mannwhitneyu(countsNorm[train][labels[train] == 0], countsNorm[train][labels[train] == 1])[1]
        pvals = np.nan_to_num(pvals, nan=1.0)
        kept = (fdrcorrection(pvals)[1] < 0.01)
        x_train = countsNorm[train][:, kept]
        # Fit classifier on train data
        model = catboost.CatBoostClassifier(iterations=100, rsm=np.sqrt(x_train.shape[1])/x_train.shape[1],
                                            class_weights=len(labels) / (2 * np.bincount(labels)), random_seed=42)
        model.fit(x_train, labels[train])
        # Predict on test data
        x_test = countsNorm[test][:, kept]
        predictions[test] = model.predict(x_test)
    with open(paths.tempDir + f"classifier_{case}.txt", "w") as f:
        print("Weighted accuracy :", balanced_accuracy_score(labels, predictions), file=f)
        wAccs[case] = balanced_accuracy_score(labels, predictions)
        print("Recall :", recall_score(labels, predictions), file=f)
        print("Precision :", precision_score(labels, predictions), file=f)
        df = pd.DataFrame(confusion_matrix(labels, predictions))
        df.columns = ["Normal Tissue True", "Tumor True"]
        df.index = ["Normal Tissue predicted", "Tumor predicted"]
        print(df, file=f)
    '''
    # DE Genes
    countsTumor = countsNorm[labels == 1]
    countsNormal = countsNorm[labels == 0]
    stat, pvals = mannwhitneyu(countsNormal, countsTumor)
    pvals = np.nan_to_num(pvals, nan=1.0)
    qvals = fdrcorrection(pvals)[1]
    meanFC = np.log2(np.percentile(countsTumor, 95, axis=0)/np.percentile(countsNormal, 95, axis=0)) 
    meanFC = np.nan_to_num(meanFC, nan=0.0)
    consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
    topLocs = consensuses.iloc[((qvals < 0.05)).nonzero()[0]]
    topLocs.to_csv(paths.tempDir + f"{case}_DE.csv", sep="\t", header=None, index=None)
    DE_upreg[case] = ((qvals < 0.05) & (meanFC > 1))
    DE_downreg[case] = ((qvals < 0.05) & (meanFC < -1))
# %%
cases = [("upreg", DE_upreg), ("downreg", DE_downreg)]
from lib.pyGREAT import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
import pyranges as pr
from lib.utils import overlap_utils
remap_catalog = pr.read_bed(paths.remapFile)
# %%
for case, mat in cases:
    countsRnd = np.zeros(mat.shape[1])
    for i in range(100):
        shuffledDF = np.apply_along_axis(np.random.permutation, 0, mat)
        counts = np.bincount(np.sum(shuffledDF, axis=1))/100
        countsRnd[:counts.shape[0]] += counts
    countsObs = np.bincount(np.sum(mat, axis=1))
    for threshold in range(len(countsObs)):
        randomSum = np.sum(countsRnd[threshold:])
        fpr = randomSum / (np.sum(countsObs[threshold:])+randomSum)
        print(threshold, fpr)
        if fpr < 0.05:
            break
    studied = np.sum(mat, axis=1) >= threshold

    consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
    globallyDE = consensuses[studied]
    globallyDE.to_csv(paths.tempDir + f"globallyDE_{case}.bed", sep="\t", header=None, index=None)


    from lib.utils import matrix_utils 
    goClass = "molecular_function"
    goEnrich = enricher.findEnriched(globallyDE, consensuses)
    print(goEnrich[goClass][2][(goEnrich[goClass][2] < 0.05) & (goEnrich[goClass][1] > 2)].sort_values()[:25])
    hasEnrich = goEnrich[goClass][2][(goEnrich[goClass][2] < 0.05) & (goEnrich[goClass][3] >= 2)]
    clustered = matrix_utils.graphClustering(enricher.matrices[goClass].loc[hasEnrich.index], "dice", 
                                            disconnection_distance=1.0, r=1.0, k=3, restarts=10)
    topK = 5
    maxP = []
    for c in np.arange(clustered.max()+1):
        inClust = hasEnrich.index[clustered == c]
        maxP.append(hasEnrich[inClust].min())
    orderedP = np.argsort(maxP)
    for c in orderedP:
        inClust = hasEnrich.index[clustered == c]
        strongest = hasEnrich[inClust].sort_values()[:topK]
        if len(strongest) > 0:
            print("-"*20)
            print(strongest)
        
    
    enrichments = overlap_utils.computeEnrichVsBg(remap_catalog, consensuses, globallyDE)
    orderedP = np.argsort(enrichments[2])
    enrichments[2][orderedP][enrichments[2][orderedP] < 0.05][:25]

# %%
