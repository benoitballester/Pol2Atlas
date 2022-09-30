# %%
import sys
import os
import numpy as np
import pandas as pd
import pyranges as pr
from lib.utils import overlap_utils

sys.path.append("./")
from settings import params, paths
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
def addAnnotation(summits, tab, colName, inputBed, showCol="Name"):
    annotBed = pr.read_bed(inputBed)
    joined = summits.join(annotBed).as_df()
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
addAnnotation(consensusesPr, intersectTab, "lncpedia", 
              paths.lncpediaPath)
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
addAnnotation(consensusesPr, intersectTab, "Fantom 5 bidirectionnal enhancers", showCol=None,
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
addAnnotation(consensusesPr, intersectTab, "ReMap CRM", showCol="Score",
              inputBed=paths.remapCrms)
# %%
intersectTab.to_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t", index=None)
# %%
try:
    os.mkdir(paths.outputDir + "intersections_databases/")
except:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
proportions = intersectTab["Encode CCREs"].value_counts()/len(intersectTab)
plt.figure(dpi=500)
sns.barplot(proportions.values, proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/encode_ccres.pdf", bbox_inches="tight")
plt.close()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
intersectLinc = intersectTab["lncpedia"].value_counts().sum()
intersectCCRE = intersectTab["Encode CCREs"].value_counts().sum()
intersectRemap = (intersectTab["ReMap CRM"] >= 10).sum()
intersectRepeat = intersectTab["Repeat Family"].value_counts().sum()
intersectF5Enh = intersectTab["Fantom 5 bidirectionnal enhancers"].sum()
intersectF5TSS = intersectTab["Fantom 5 TSSs"].sum()
intersectDnase = intersectTab["DNase meuleman"].value_counts().sum()
df = pd.DataFrame([intersectLinc, intersectCCRE, intersectRemap, intersectRepeat, intersectF5Enh,
                   intersectF5TSS, intersectDnase], index=["lncPedia", "Encode CCREs", "ReMap CRMs", "Repeated elements", "F5 enhancers", "F5 TSSs", "ENCODE Dnase"])
df /= len(consensuses)
df.sort_values(0, inplace=True, ascending=False)
df["Cat"] = df.index
plt.figure(dpi=500)
sns.barplot(x=df[0], y=df.index)
plt.xticks(rotation=90)
plt.ylabel("Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/prop_all_db.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
proportions = (intersectTab["Repeat Class"].value_counts()/len(intersectTab))[:10]
plt.figure(dpi=500)
sns.barplot(proportions.values, proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/rep_family.pdf", bbox_inches="tight")
plt.close()
# %%
