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
from lib import mlp
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import catboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, ttest_ind, ranksums
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA
# %%
allAnnots = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", 
                        sep="\t", index_col=0)
DE_Idx = pd.DataFrame()
wAccs = pd.DataFrame()
for case in allAnnots["project_id"].unique():
    case = "TCGA-PRAD"
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
    labels = []
    for a in annotation["Sample Type"]:
        if a == "Solid Tissue Normal":
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)
    if len(labels) < 10 or np.any(np.bincount(labels) < 10):
        print(case, "not enough samples")
        continue
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
    allCounts = np.concatenate(counts, axis=1)

    # Normalize and remove 0 counts
    scale = np.mean(allCounts, axis=0)
    countsNorm = allCounts.T / allReads[:, None]
    countsNorm = countsNorm / np.min(countsNorm[countsNorm.nonzero()])
    nz = np.sum(countsNorm >= 1, axis=0) >= 3
    countsNorm = countsNorm[:, nz]
    # Predictive model
    labels = []
    for a in annotation["Sample Type"].loc[order]:
        if a == "Solid Tissue Normal":
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)
    labels_cat = tf.keras.utils.to_categorical(labels)
    print(labels_cat)
    predictions = np.zeros(len(labels), dtype=int)
    for train, test in StratifiedKFold(10, shuffle=True, random_state=42).split(countsNorm, labels):
        x_train = countsNorm[train]
        # Fit classifier on train data
        model = mlp.mlp(x_train.shape, 2)
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels[train]),
                                                 labels[train])
        weights = class_weights[labels[train]]
        model.fit(x_train, labels_cat[train], epochs=100, sample_weight=weights)
        # Predict on test data
        x_test = countsNorm[test]
        predictions[test] = np.argmax(model.predict(x_test), axis=1)

    print("Weighted accuracy :", balanced_accuracy_score(labels, predictions))
    wAccs[case] = balanced_accuracy_score(labels, predictions)
    print("Recall :", recall_score(labels, predictions))
    print("Precision :", precision_score(labels, predictions))
    df = pd.DataFrame(confusion_matrix(labels, predictions))
    df.columns = ["Normal Tissue True", "Tumor True"]
    df.index = ["Normal Tissue predicted", "Tumor predicted"]
    print(df)
    break
# %%
