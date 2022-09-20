# %%
import sys

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
