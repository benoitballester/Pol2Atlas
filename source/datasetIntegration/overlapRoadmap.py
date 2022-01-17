# %%
import pyranges as pr
import pandas as pd
from settings import params, paths
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

try:
    os.mkdir(paths.outputDir + "epigenetic/")
except FileExistsError:
    pass
# %%
# Load consensuses and clusters
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensusesPr = consensuses[[0,6,7]]
consensusesPr["Name"] = np.arange(len(consensusesPr))
consensusesPr.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensusesPr)
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)

# Find roadmap files
roadmapPath = "/scratch/pdelangen/projet_these/data_clean/roadmap_chromatin_states/"
roadmapBeds = os.listdir(roadmapPath)
roadmapBeds = [f for f in roadmapBeds if f.endswith(".bed.gz")]
epigenomes = [f.split("_")[0] for f in roadmapBeds]
consensusEpigenomeMat = np.zeros((len(consensuses), len(roadmapBeds)), dtype="int16")
epigenomeMetadata = pd.read_csv(roadmapPath + "Roadmap.metadata.qc.jul2013 - Consolidated_EpigenomeIDs_summary_Table.csv",
                                usecols=["Epigenome ID (EID)", "GROUP"])
epigenomeMetadata.dropna(inplace=True)
epigenomeMetadata.set_index("Epigenome ID (EID)", inplace=True)
epigenomeMetadata = epigenomeMetadata.loc[epigenomes]

# %%
# For each consensus peak, for each epigenome, get the Roadmap chromHmm annotation
for i in range(len(roadmapBeds)):
    epigenome = pd.read_csv(roadmapPath + roadmapBeds[i], header=None, sep="\t")
    try:
        chrStateMap
    except:
        chrStateMap = LabelEncoder()
        chrStateMap.fit(epigenome[3])
    epigenome[3] = chrStateMap.transform(epigenome[3])+1
    epigenome.columns = ["Chromosome", "Start", "End", "Name"]
    epigenome = pr.PyRanges(epigenome)
    joined = consensusesPr.join(epigenome).as_df()
    consensusEpigenomeMat[joined["Name"], i] = joined["Name_b"]
# %%
# All Pol II
def computeProps(countsMat, epigenomes, classes, targetCol):
    countsPerEpigenome = np.array([np.bincount(c, minlength=16) for c in countsMat.T])[:, 1:]
    proportionsPerEpigenome = countsPerEpigenome / np.sum(countsPerEpigenome, axis=1)[:, None]
    props = proportionsPerEpigenome.ravel()
    counts = countsPerEpigenome.ravel()
    cat = np.tile(classes, proportionsPerEpigenome.shape[0])
    epigenomesID = np.repeat(epigenomes, proportionsPerEpigenome.shape[1])
    df = pd.DataFrame(np.transpose([props, cat, epigenomesID]), columns=["Proportions", "Category", "Epigenome"])
    df["Targets"] = targetCol
    return df

df = computeProps(consensusEpigenomeMat, epigenomes, chrStateMap.classes_, targetCol="All Pol II and all epigenomes")
# %%
def enrichForClust(clust, roadmapGroup, name, consensusEpigenomeMat, epigenomeMetadata, chrStateMap):
    usedEpigenomes = (epigenomeMetadata["GROUP"] == roadmapGroup)
    df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==clust)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes")
    df2 = computeProps(consensusEpigenomeMat[clusts==clust][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'{name}' cluster, in '{roadmapGroup}' epigenomes")
    dfAll = pd.concat([df1, df2])
    df1PerCat = dict([(g, x) for g, x in df1.groupby("Category")])
    df2PerCat = dict([(g, x) for g, x in df2.groupby("Category")])


    order = ["1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk", "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv", "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies"]
    plt.figure(figsize=(8,4.5), dpi=500)
    sns.boxplot(x="Category", y="Proportions", data=dfAll, hue="Targets", dodge=True, showfliers=False, order=order)
    plt.gca().legend_ = None
    sns.stripplot(x="Category", y="Proportions", data=dfAll, hue="Targets", jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2, linewidth=0.1, order=order)
    # Show statistical significance
    for k in df1PerCat.keys():
        epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        stat, p = ttest_rel(epigenomes1, epigenomes2)
        pos = order.index(k)
        sig = 0
        if p < 0.05:
            sig = min(int(-np.log10(p)), 4)
        medianDiff = np.median(epigenomes2) - np.median(epigenomes1)
        maxVal = max(epigenomes1.max(), epigenomes2.max())
        txt = "- "
        if medianDiff > 0:
            txt = "+"
        plt.text(pos, maxVal+0.01, sig*txt, ha="center", fontsize=8, fontweight="heavy")
    plt.legend(fontsize=8)
    plt.xticks(rotation=70)
    plt.savefig(paths.outputDir + "epigenetic/" + f"clust_{name}_roadmap_{roadmapGroup}.pdf", bbox_inches="tight")
    plt.figure()
    plt.show()
    plt.close()
    
enrichForClust(4, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClust(6, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClust(9, "Heart", "Cardiovascular", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClust(18, "Brain", "Nervous", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
