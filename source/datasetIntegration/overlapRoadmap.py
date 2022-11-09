# %%
import pyranges as pr
import pandas as pd
import os
import sys
sys.path.append("./")
from settings import params, paths
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, chi2_contingency

try:
    os.mkdir(paths.outputDir + "epigenetic/")
except FileExistsError:
    pass
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
# %%
# Load consensuses and clusters
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensusesPr = consensuses[[0,6,7]]
consensusesPr["Name"] = np.arange(len(consensusesPr))
consensusesPr.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensusesPr)
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)

# Find roadmap files
roadmapPath = paths.roadmapPath
roadmapBeds = os.listdir(roadmapPath)
roadmapBeds = [f for f in roadmapBeds if f.endswith(".bed.gz")]
epigenomes = [f.split("_")[0] for f in roadmapBeds]
consensusEpigenomeMat = np.zeros((len(consensuses), len(roadmapBeds)), dtype="int16")
epigenomeMetadata = pd.read_csv(roadmapPath + "Roadmap.metadata.qc.jul2013 - Consolidated_EpigenomeIDs_summary_Table.csv",
                                usecols=["Epigenome ID (EID)", "GROUP"])
epigenomeMetadata.dropna(inplace=True)
epigenomeMetadata.set_index("Epigenome ID (EID)", inplace=True)
epigenomeMetadata = epigenomeMetadata.loc[epigenomes]
color = pd.read_csv(roadmapPath + "Roadmap.metadata.qc.jul2013 - Consolidated_EpigenomeIDs_summary_Table.csv",
                                usecols=["GROUP", "COLOR"])
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
    df["Overlaps"] = countsPerEpigenome.ravel()
    df["Total overlaps"] = np.repeat(np.sum(countsPerEpigenome, axis=1), proportionsPerEpigenome.shape[1])
    return df

df = computeProps(consensusEpigenomeMat, epigenomes, chrStateMap.classes_, 
                  targetCol="All Pol II and all epigenomes")
# %%
# Case specific, cluster specific
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
        # epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        # epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        prop1 = df1PerCat[k]["Proportions"].values.astype("float")
        prop2 = df2PerCat[k]["Proportions"].values.astype("float")
        try:
            # p = chi2_contingency(np.array([epigenomes1, epigenomes2]).T)[1]
            p = ttest_rel(prop1, prop2)[1]
        # If full zero counts are detected it raises an error
        # p is equal to 1 in this case
        except ValueError:
            p = 1.0
        pos = order.index(k)
        sig = 0
        if p < 0.05:
            sig = min(int(-np.log10(p+1e-300)), 4)
        
        meanDiff = np.mean(prop2) - np.mean(prop1)
        maxVal = max(prop1.max(), prop2.max())
        txt = "- "
        if meanDiff > 0:
            txt = "+"
        plt.text(pos, maxVal+0.01, sig*txt, ha="center", fontsize=8, fontweight="heavy")
    plt.legend(fontsize=8)
    plt.legend([],[], frameon=False)
    plt.xticks(rotation=70)
    plt.savefig(paths.outputDir + "epigenetic/" + f"clust_{name}_roadmap_{roadmapGroup}.pdf", bbox_inches="tight")
    plt.figure()
    plt.show()
    plt.close()
    
enrichForClust(5, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClust(4, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
# Case specific, cluster specific, violin
def enrichForClust(clust, roadmapGroup, name, consensusEpigenomeMat, epigenomeMetadata, chrStateMap):
    usedEpigenomes = (epigenomeMetadata["GROUP"] == roadmapGroup)
    df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==clust)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes")
    df2 = computeProps(consensusEpigenomeMat[clusts==clust][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'{name}' cluster, in '{roadmapGroup}' epigenomes")
    dfAll = pd.concat([df1, df2])
    dfAll.index = np.arange(len(dfAll))
    df1PerCat = dict([(g, x) for g, x in df1.groupby("Category")])
    df2PerCat = dict([(g, x) for g, x in df2.groupby("Category")])
    palette = {f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes": (0.9,0.9,0.9), 
               f"'{name}' cluster, in '{roadmapGroup}' epigenomes":sns.color_palette()[1]}
    dfAll["Category"] = dfAll["Category"].astype("category")
    dfAll["Proportions"] = dfAll["Proportions"].astype("float")

    order = ["1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk", "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv", "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies"]
    plt.figure(figsize=(3.5,2), dpi=500)
    sns.violinplot(x="Category", y="Proportions", data=dfAll, hue="Targets", dodge=True, 
                showfliers=False, order=order, linewidth=0.2, linecolor="k")
    plt.gca().legend_ = None
    """ sns.stripplot(x="Category", y="Proportions", data=dfAll, hue="Targets", jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2, linewidth=0.1, order=order) """
    # Show statistical significance
    for k in df1PerCat.keys():
        # epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        # epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        prop1 = df1PerCat[k]["Proportions"].values.astype("float")
        prop2 = df2PerCat[k]["Proportions"].values.astype("float")
        try:
            # p = chi2_contingency(np.array([epigenomes1, epigenomes2]).T)[1]
            p = ttest_rel(prop1, prop2)[1]
        # If full zero counts are detected it raises an error
        # p is equal to 1 in this case
        except ValueError:
            p = 1.0
        pos = order.index(k)
        sig = 0
        if p < 0.05:
            sig = min(int(-np.log10(p+1e-300)), 3)
        
        meanDiff = np.mean(prop2) - np.mean(prop1)
        maxVal = max(prop1.max(), prop2.max())
        txt = "-"
        if meanDiff > 0:
            txt = "+"
        plt.text(pos, maxVal+0.02, sig*txt, ha="center", fontsize=4, fontweight="heavy")
    plt.legend(fontsize=8)
    plt.xlim((-0.5,14.5))
    plt.ylim(plt.ylim()[0], plt.ylim()[1])
    plt.vlines(np.arange(1,15)-0.5, plt.ylim()[0], plt.ylim()[1], 
               linewidths=0.5, linestyle=(5, (5, 10)), color="grey", alpha=0.2)
    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel("Proportions", fontsize=10)
    plt.savefig(paths.outputDir + "epigenetic/" + f"violin_clust_{name}_roadmap_{roadmapGroup}.pdf", 
                bbox_inches="tight")
    plt.show()
    plt.close()
enrichForClust(5, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClust(4, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
# Case specific, cluster specific, violin horizontal
def enrichForClustHor(clust, roadmapGroup, name, consensusEpigenomeMat, epigenomeMetadata, chrStateMap):
    usedEpigenomes = (epigenomeMetadata["GROUP"] == roadmapGroup)
    df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==clust)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes")
    df2 = computeProps(consensusEpigenomeMat[clusts==clust][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'{name}' cluster, in '{roadmapGroup}' epigenomes")
    dfAll = pd.concat([df1, df2])
    dfAll.index = np.arange(len(dfAll))
    df1PerCat = dict([(g, x) for g, x in df1.groupby("Category")])
    df2PerCat = dict([(g, x) for g, x in df2.groupby("Category")])
    palette = {f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes": (0.9,0.9,0.9), 
               f"'{name}' cluster, in '{roadmapGroup}' epigenomes":sns.color_palette()[1]}
    dfAll["Category"] = dfAll["Category"].astype("category")
    dfAll["Proportions"] = dfAll["Proportions"].astype("float")

    order = ["1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk", "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv", "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies"]
    plt.figure(figsize=(2.75,2.5), dpi=500)
    sns.boxenplot(y="Category", x="Proportions", data=dfAll, hue="Targets", dodge=True,
                showfliers=False, order=order[::-1], k_depth="full", linewidth=0.4, width=0.95,
                flier_kws={"marker": "x", "s":3, "linewidths":0.2, "color":"k"}
                ) 
    """
    sns.violinplot(y="Category", x="Proportions", data=dfAll, hue="Targets", dodge=True,
                showfliers=False, order=order[::-1], scale="width", linewidth=0.2, width=0.95, 
                )""" 
    """ 'whiskerprops':{'color':'blue'},
    'capprops':{'color':'yellow'} 
    sns.stripplot(y="Category", x="Proportions", data=dfAll, hue="Targets", jitter=0.33, dodge=True, 
                edgecolor="black",alpha=1.0, s=1, linewidth=0.1, order=order[::-1])"""
    plt.gca().legend_ = None
    """ sns.stripplot(x="Category", y="Proportions", data=dfAll, hue="Targets", jitter=0.33, dodge=True, 
                    edgecolor="black",alpha=1.0, s=2, linewidth=0.1, order=order) """
    # Show statistical significance
    for k in df1PerCat.keys():
        # epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        # epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        prop1 = df1PerCat[k]["Proportions"].values.astype("float")
        prop2 = df2PerCat[k]["Proportions"].values.astype("float")
        try:
            # p = chi2_contingency(np.array([epigenomes1, epigenomes2]).T)[1]
            p = ttest_rel(prop1, prop2)[1]
        # If full zero counts are detected it raises an error
        # p is equal to 1 in this case
        except ValueError:
            p = 1.0
        pos = order[::-1].index(k)
        sig = 0
        if p < 0.05:
            sig = min(int(-np.log10(p+1e-300)), 3)
        
        meanDiff = np.mean(prop2) - np.mean(prop1)
        maxVal = max(prop1.max(), prop2.max())
        txt = "- "
        if meanDiff > 0:
            txt = "+"
        print(k, pos-1)
        plt.text(maxVal+0.1, pos, sig*txt, va="center", ha="center", fontsize=4, fontweight="heavy", color="red")
    plt.legend(fontsize=8)
    plt.legend([],[], frameon=False)
    plt.ylim((-0.5,14.5))
    plt.xlim(0, plt.xlim()[1])
    plt.hlines(np.arange(1,15)-0.5, plt.xlim()[0], plt.xlim()[1], 
               linewidths=0.5, linestyle=(0, (5, 5)), color="k", alpha=0.3)
    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    plt.gca().tick_params(axis=u'y', which=u'both',length=0)
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.ylabel("Epigenetic state", fontsize=6)
    plt.xlabel("Proportion", fontsize=6)
    plt.savefig(paths.outputDir + "epigenetic/" + f"boxen_hor_clust_{name}_roadmap_{roadmapGroup}.pdf", 
                bbox_inches="tight")
    plt.show()
    plt.close()
enrichForClustHor(5, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClustHor(4, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
# Case specific, cluster specific, violin horizontal
def enrichForClustBar(clust, roadmapGroup, name, consensusEpigenomeMat, epigenomeMetadata, chrStateMap):
    usedEpigenomes = (epigenomeMetadata["GROUP"] == roadmapGroup)
    df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==clust)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes")
    df2 = computeProps(consensusEpigenomeMat[clusts==clust][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'{name}' cluster, in '{roadmapGroup}' epigenomes")
    dfAll = pd.concat([df1, df2])
    dfAll.index = np.arange(len(dfAll))
    df1PerCat = dict([(g, x) for g, x in df1.groupby("Category")])
    df2PerCat = dict([(g, x) for g, x in df2.groupby("Category")])
    palette = {f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes": (0.9,0.9,0.9), 
               f"'{name}' cluster, in '{roadmapGroup}' epigenomes":sns.color_palette()[1]}
    dfAll["Category"] = dfAll["Category"].astype("category")
    dfAll["Proportions"] = dfAll["Proportions"].astype("float")

    order = ["1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk", "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv", "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies"]
    plt.figure(figsize=(3.5,3.5), dpi=500)
    plt.gca().legend_ = None

    # Show statistical significance
    allP = []
    for k in df1PerCat.keys():
        # epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        # epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        prop1 = df1PerCat[k]["Proportions"].values.astype("float")
        prop2 = df2PerCat[k]["Proportions"].values.astype("float")
        try:
            # p = chi2_contingency(np.array([epigenomes1, epigenomes2]).T)[1]
            p = ttest_rel(prop1, prop2)[1]
        # If full zero counts are detected it raises an error
        # p is equal to 1 in this case
        except ValueError:
            p = 1.0
        pos = order[::-1].index(k)
        sig = 0
        if p < 0.05:
            sig = min(int(-np.log10(p+1e-300)), 3)
        meanDiff = np.mean(prop2) - np.mean(prop1)
        maxVal = max(prop1.max(), prop2.max())
        allP.append(-np.log10(p+1e-300))
        plt.barh(pos, -np.log10(p+1e-300) * np.sign(meanDiff))
        txt = "- "
        if meanDiff > 0:
            txt = "+"
        print(k, pos)
        # plt.text(0, pos, sig*txt, va="center", ha="center", fontsize=6, fontweight="heavy")
    plt.legend(fontsize=8)
    plt.legend([],[], frameon=False)
    plt.ylim((-0.5,14.5))
    print(allP)
    # plt.xlim(-np.percentile(allP, 95)*1.1, np.percentile(allP, 95)*1.1)
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.hlines(np.arange(1,15)-0.5, plt.xlim()[0], plt.xlim()[1], 
               linewidths=0.5, linestyle=(5, (5, 10)), color="grey", alpha=0.2)
    plt.vlines([0.0], plt.ylim()[0], plt.ylim()[1], color="grey")
    plt.vlines([-np.log10(0.05), np.log10(0.05)], plt.ylim()[0], plt.ylim()[1], 
                linestyle="dashed", color="red")
    plt.text(np.log10(0.05), plt.ylim()[1]+0.5, "p=0.05", va="center", ha="center", fontsize=6, color="red")
    plt.text(-np.log10(0.05), plt.ylim()[1]+0.5, "p=0.05", va="center", ha="center", fontsize=6, color="red")
    plt.tight_layout()
    plt.xticks(fontsize=8)
    plt.yticks(np.arange(len(order)), order[::-1], fontsize=8)
    plt.ylabel("Chromatin state", fontsize=10)
    plt.xlabel("-log10(p-value) * sign of enrichment", fontsize=10)
    plt.savefig(paths.outputDir + "epigenetic/" + f"enrichBarplot_{name}_roadmap_{roadmapGroup}.pdf", 
                bbox_inches="tight")
    plt.show()
    plt.close()
enrichForClustBar(5, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClustBar(4, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
# Case specific, cluster specific, bubbleplot
def enrichForClustBubble(clust, roadmapGroup, name, consensusEpigenomeMat, epigenomeMetadata, chrStateMap):
    usedEpigenomes = (epigenomeMetadata["GROUP"] == roadmapGroup)
    df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==clust)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes")
    df2 = computeProps(consensusEpigenomeMat[clusts==clust][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'{name}' cluster, in '{roadmapGroup}' epigenomes")
    dfAll = pd.concat([df1, df2])
    dfAll.index = np.arange(len(dfAll))
    df1PerCat = dict([(g, x) for g, x in df1.groupby("Category")])
    df2PerCat = dict([(g, x) for g, x in df2.groupby("Category")])
    palette = {f"Not '{name}' cluster, in '{roadmapGroup}' epigenomes": (0.9,0.9,0.9), 
               f"'{name}' cluster, in '{roadmapGroup}' epigenomes":sns.color_palette()[1]}
    dfAll["Category"] = dfAll["Category"].astype("category")
    dfAll["Proportions"] = dfAll["Proportions"].astype("float")

    order = ["1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk", "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv", "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies"]
    plt.figure(figsize=(3.5,3.5), dpi=500)
    plt.gca().legend_ = None
    colormap = sns.color_palette("vlag", as_cmap=True)
    # Show statistical significance
    allP = []
    allLfc = []
    for k in df1PerCat.keys():
        # epigenomes1 = df1PerCat[k]["Proportions"].values.astype("float")
        # epigenomes2 = df2PerCat[k]["Proportions"].values.astype("float")
        prop1 = df1PerCat[k]["Proportions"].values.astype("float")
        prop2 = df2PerCat[k]["Proportions"].values.astype("float")
        try:
            # p = chi2_contingency(np.array([epigenomes1, epigenomes2]).T)[1]
            p = ttest_rel(prop1, prop2)[1]
        # If full zero counts are detected it raises an error
        # p is equal to 1 in this case
        except ValueError:
            p = 1.0
        pos = order[::-1].index(k)

        allP.append(-np.log10(p+1e-300))
        lfc = np.clip(np.log2(np.mean(prop2)/np.mean(prop1)),-1,1) * 0.5 + 0.5
        c = colormap(lfc)
        if p > 0.05:
            c = (0.1,0.1,0.1)
        plt.scatter(np.mean(prop2),pos,  s=np.square(-np.log10(p+1e-300))*1.8, c=c, linewidths=0.0)
        print(k, pos)
    plt.legend(fontsize=8)
    plt.legend([],[], frameon=False)
    plt.ylim((-0.5,14.5))
    print(allP)
    # plt.xlim(-np.percentile(allP, 95)*1.1, np.percentile(allP, 95)*1.1)
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.hlines(np.arange(1,15)-0.5, plt.xlim()[0], plt.xlim()[1], 
               linewidths=0.5, linestyle=(5, (5, 10)), color="grey", alpha=0.2)
    plt.tight_layout()
    plt.xticks(fontsize=8)
    plt.yticks(np.arange(len(order)), order[::-1], fontsize=8)
    plt.ylabel("Chromatin state", fontsize=10)
    plt.xlabel(f"Proportion in {name} cluster, in {roadmapGroup} epigenomes", fontsize=10)
    plt.savefig(paths.outputDir + "epigenetic/" + f"enrichBubble_{name}_roadmap_{roadmapGroup}.pdf", 
                bbox_inches="tight")
    plt.show()
    plt.close()
enrichForClustBubble(5, "Blood & T-cell", "Lymphoid", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
enrichForClustBubble(4, "ESC", "Embryonic", consensusEpigenomeMat, epigenomeMetadata, chrStateMap)
# %%
usedEpigenomes = epigenomeMetadata["GROUP"] == "ESC"
df1 = computeProps(consensusEpigenomeMat[np.logical_not(clusts==9)][:, (usedEpigenomes)], 
                    np.array(epigenomes)[(usedEpigenomes)], chrStateMap.classes_, 
                    targetCol=f"Not 'embryo' cluster, in 'ESC' epigenomes")
df2 = computeProps(consensusEpigenomeMat[clusts==9][:, usedEpigenomes], np.array(epigenomes)[usedEpigenomes], 
                    chrStateMap.classes_, targetCol=f"'Embryo' cluster, in 'ESC' epigenomes")
# %%
