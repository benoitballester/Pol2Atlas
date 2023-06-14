# %%
import sys
sys.path.append("./")
from settings import params, paths
import os
import numpy as np
import pandas as pd
import pyranges as pr
from lib.utils import overlap_utils
from sklearn.preprocessing import LabelEncoder

consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
intersectTab = consensuses[[0,1,2,3,4]]
intersectTab.columns = ["Chromosome", "Start", "End", "Name", "# experiments"]
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)
intersectTab["Cluster Id"] = clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)
# Summits only
consensusesPr = consensuses[[0,6,7]]
consensusesPr["Name"] = np.arange(len(consensusesPr))
consensusesPr.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensusesPr)
# %%
def addAnnotation(summits, tab, colName, inputBed, showCol="Name", slack=0):
    annotBed = pr.read_bed(inputBed)
    joined = summits.join(annotBed, slack=slack).as_df()
    if showCol is not None:
        tab.loc[:, colName] = None
        try:
            tab.loc[joined["Name"].values, colName] = joined[showCol + "_b"].values
        except KeyError:
            tab.loc[joined["Name"].values, colName] = joined[showCol].values
    else:
        tab.loc[:, colName] = 0
        tab.loc[joined["Name"].values, colName] = 1
# %%
# Encode CCREs
addAnnotation(consensusesPr, intersectTab, "Encode CCREs", 
              paths.ccrePath)
# %%
# lncpedia
addAnnotation(consensusesPr, intersectTab, "LNCipedia", 
              paths.lncpediaPath, slack=1000)
# %%
# Repeat Family
addAnnotation(consensusesPr, intersectTab, "Repeat Family", 
              paths.repeatFamilyBed)
# %%
# Repeat Class
addAnnotation(consensusesPr, intersectTab, "Repeat Class", 
              paths.repeatClassBed)
# %%
# Repeat Type
addAnnotation(consensusesPr, intersectTab, "Repeat Type", 
              paths.repeatTypeBed)              
# %%
# Fantom eRNA
addAnnotation(consensusesPr, intersectTab, "Fantom 5 Enhancers", showCol=None,
              inputBed=paths.f5Enh)
# %%
# Fantom TSS
addAnnotation(consensusesPr, intersectTab, "Fantom 5 TSSs", showCol=None,
              inputBed=paths.f5Cage)
# %%
# Dnase meuleman
addAnnotation(consensusesPr, intersectTab, "DNase meuleman", 
              inputBed=paths.dnaseMeuleman)
# %%
# CRM remap
addAnnotation(consensusesPr, intersectTab, "ReMap CRMs", showCol="Score",
              inputBed=paths.remapCrms)
# %%
# STARR-seq
starrseq = pr.read_bed(paths.dataFolder+"genome_annotation/ENCODE_narrowpeak_sorted_hg38.narrowPeak")
has_starr = consensusesPr.overlap(starrseq).as_df()["Name"]
intersectTab["ENCODE STARR-seq"] = False
intersectTab.loc[has_starr, "ENCODE STARR-seq"] = True
# %%
# LNCipedia promoters
lnc = pr.read_bed(paths.lncpediaPath, as_df=True)
tss = lnc["Start"].where(lnc["Strand"]=="+", lnc["End"]+1)
lncProms = lnc[["Chromosome", "Start", "End", "Name"]].copy()
lncProms["Start"] = np.maximum(tss - 1000, 0) 
lncProms["End"] = tss + 1000 
joined = consensusesPr.join(pr.PyRanges(lncProms)).as_df()
intersectTab.loc[:, "LNCipedia Promoter"] = None
try:
    intersectTab.loc[joined["Name"].values, "LNCipedia Promoter"] = joined["Name" + "_b"].values
except KeyError:
    intersectTab.loc[joined["Name"].values, "LNCipedia Promoter"] = joined["Name"].values

# %%
intersectTab.to_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t", index=None)
genomeSize = pd.read_csv(paths.genomeFile, sep="\t", header=None)[1].sum()
intergenicBed = pr.read_bed(paths.tempDir + "intergenicRegions_gc38.bed")
# %%
ccreInterg = pr.read_bed(paths.ccrePath).intersect(intergenicBed, how="containment")
ccreInterg = ccreInterg.as_df()
ccreInterg["Size"] = ccreInterg["End"]-ccreInterg["Start"]
intergData = intergenicBed.as_df()
intergData["Size"] = intergData["End"]-intergData["Start"]
mapping = {"pELS,CTCF-bound":"Proximal enhancer",
           "pELS":"Proximal enhancer",
           "dELS":"Distal enhancer",
           "dELS,CTCF-bound":"Distal enhancer",
           "PLS,CTCF-bound":"Promoter like",
           "PLS":"Promoter like",
           "CTCF-only,CTCF-bound":"CTCF",
           "DNase-H3K4me3,CTCF-bound":"DNase-H3K4me3",
           "DNase-H3K4me3":"DNase-H3K4me3",
           }
ccreInterg.replace(mapping, inplace=True)
coverage = ccreInterg.groupby(["Name"])["Size"].sum()
intergSize = intergData["Size"].sum()
prob = coverage / intergSize
prob["No overlap"] = 1-prob.sum()
try:
    os.mkdir(paths.outputDir + "intersections_databases/")
except:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import binom
from statsmodels.stats.multitest import fdrcorrection
proportions = intersectTab["Encode CCREs"].replace(mapping).value_counts()
proportions["No overlap"] = len(intersectTab) - proportions.sum()
proportions.sort_values(inplace=True, ascending=False)
# Two sided test
pvals = binom(len(consensuses), prob.loc[proportions.index]).sf(proportions-1)
pvals = np.minimum(pvals, binom(len(consensuses), prob.loc[proportions.index]).cdf(proportions-1))*2.0
sig = fdrcorrection(pvals)[0]
plt.figure(dpi=500)
sns.barplot(x=proportions.values/len(intersectTab), y=proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(len(proportions)):
    fc = np.around(proportions[i]/len(intersectTab)/prob.loc[proportions.index][i], 2)
    if sig[i]:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
plt.savefig(paths.outputDir + "intersections_databases/encode_ccres_lowcat.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
def computeSubsetProb(bedPath, intergRegions):
    ccreInterg = pr.read_bed(bedPath).intersect(intergRegions, how="containment")
    ccreInterg = ccreInterg.as_df()
    ccreInterg["Size"] = ccreInterg["End"]-ccreInterg["Start"]
    intergData = intergRegions.as_df()
    intergData["Size"] = intergData["End"]-intergData["Start"]
    coverage = ccreInterg.groupby(["Name"])["Size"].sum()
    intergSize = intergData["Size"].sum()
    print(intergSize)
    return coverage / intergSize, coverage
prob, coverage = computeSubsetProb(paths.ccrePath, intergenicBed)
try:
    os.mkdir(paths.outputDir + "intersections_databases/")
except:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import binom
from statsmodels.stats.multitest import fdrcorrection
proportions = intersectTab["Encode CCREs"].value_counts()
proportions.sort_values(inplace=True, ascending=False)
pvals = binom(len(consensuses), prob.loc[proportions.index]).sf(proportions-1)
pvals = np.minimum(pvals, binom(len(consensuses), prob.loc[proportions.index]).cdf(proportions-1))*2.0
sig = fdrcorrection(pvals)[0]
plt.figure(dpi=500)
sns.barplot(x=proportions.values/len(intersectTab), y=proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(len(proportions)):
    fc = np.around(proportions[i]/len(intersectTab)/prob.loc[proportions.index][i], 2)
    if sig[i]:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
plt.savefig(paths.outputDir + "intersections_databases/encode_ccres.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
def computeSubsetProbNoCat(bedPath, intergRegions, minScore=None):
    ccreInterg = pr.read_bed(bedPath).intersect(intergRegions, how="containment")
    ccreInterg = ccreInterg.as_df()
    ccreInterg["Size"] = ccreInterg["End"]-ccreInterg["Start"]
    intergData = intergRegions.as_df()
    intergData["Size"] = intergData["End"]-intergData["Start"]
    if minScore is not None:
        ccreInterg = ccreInterg[ccreInterg["Score"] >= 10]
    coverage = ccreInterg["Size"].sum()
    intergSize = intergData["Size"].sum()
    return coverage / intergSize
probLinc = computeSubsetProbNoCat(paths.lncpediaPath, intergenicBed)
intersectLinc = intersectTab["LNCipedia"].value_counts().sum()
probCCRE = computeSubsetProbNoCat(paths.ccrePath, intergenicBed)
intersectCCRE = intersectTab["Encode CCREs"].value_counts().sum()
probRemap = computeSubsetProbNoCat(paths.remapCrms, intergenicBed, 10)
intersectRemap = (intersectTab["ReMap CRMs"] >= 10).sum()
probRepeat = computeSubsetProbNoCat(paths.repeatFamilyBed, intergenicBed)
intersectRepeat = intersectTab["Repeat Family"].value_counts().sum()
probF5Enh = computeSubsetProbNoCat(paths.f5Enh, intergenicBed)
intersectF5Enh = intersectTab["Fantom 5 Enhancers"].sum()
probF5TSS = computeSubsetProbNoCat(paths.f5Cage, intergenicBed)
intersectF5TSS = intersectTab["Fantom 5 TSSs"].sum()
probDnase = computeSubsetProbNoCat(paths.dnaseMeuleman, intergenicBed)
intersectDnase = intersectTab["DNase meuleman"].value_counts().sum()
df = pd.DataFrame([intersectLinc, intersectCCRE, intersectRemap, intersectRepeat, intersectF5Enh,
                   intersectF5TSS, intersectDnase], index=["lncPedia", "Encode CCREs", "ReMap CRMs", "Repeated elements", "F5 enhancers", "F5 TSSs", "ENCODE Dnase"])
df["Probs"] = [probLinc, probCCRE, probRemap, probRepeat, probF5Enh,
                   probF5TSS, probDnase]
df.sort_values(0, inplace=True, ascending=False)
df["Cat"] = df.index
pvals = binom(len(consensuses), df["Probs"]).sf(df[0]-1)
pvals = np.minimum(pvals, binom(len(consensuses), df["Probs"]).cdf(df[0]-1))*2.0
sig = fdrcorrection(pvals)[0]
plt.figure(dpi=500)
sns.barplot(x=df[0]/len(consensuses), y=df.index)
plt.xticks(rotation=90)
plt.ylabel("Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(len(df)):
    fc = np.around(df.iloc[i, 0]/len(intersectTab)/df['Probs'][i], 2)
    if sig[i]:
        plt.text(df.iloc[i, 0]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(df.iloc[i, 0]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
print(pvals)
plt.savefig(paths.outputDir + "intersections_databases/prop_all_db.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
prob, coverage = computeSubsetProb(paths.repeatClassBed, intergenicBed)
proportions = (intersectTab["Repeat Class"].value_counts())[:10]
# Two sided test
pvals = binom(len(consensuses), prob.loc[proportions.index]).sf(proportions-1)
pvals = np.minimum(pvals, binom(len(consensuses), prob.loc[proportions.index]).cdf(proportions-1))*2.0
sig = fdrcorrection(pvals)[0]
plt.figure(dpi=500)
sns.barplot(x=proportions.values/len(intersectTab), y=proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(len(proportions)):
    fc = np.around(proportions[i]/len(intersectTab)/prob.loc[proportions.index][i], 2)
    if sig[i]:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
plt.savefig(paths.outputDir + "intersections_databases/rep_family.pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
