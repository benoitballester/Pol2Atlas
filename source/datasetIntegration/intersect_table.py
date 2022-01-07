# %%
from lib.utils import overlap_utils
import numpy as np
import pandas as pd
import pyranges as pr
from sklearn.preprocessing import LabelEncoder
from settings import params, paths

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
        intersectTab.loc[:, colName] = None
        intersectTab.loc[joined["Name"].values, colName] = joined[showCol + "_b"].values
    else:
        intersectTab.loc[:, colName] = 0
        intersectTab.loc[joined["Name"].values, colName] = 1
# %%
# Encode CCREs
addAnnotation(consensusesPr, intersectTab, "Encode CCREs", 
              "/scratch/pdelangen/projet_these/data/annotation/GRCh38-ccREsFix.bed")
# %%
# lncpedia
addAnnotation(consensusesPr, intersectTab, "lncpedia", 
              "/scratch/pdelangen/projet_these/data/annotation/lncipedia_5_2_hg38.bed")
# %%
# Repeat Family
addAnnotation(consensusesPr, intersectTab, "Repeat Family", 
              "/scratch/pdelangen/projet_these/oldBackup/temp_POL2_lenient/repeatBedFamily.bed")
# %%
# Repeat Class
addAnnotation(consensusesPr, intersectTab, "Repeat Class", 
              "/scratch/pdelangen/projet_these/oldBackup/temp_POL2_lenient/repeatBedClass.bed")
# %%
# Repeat Type
addAnnotation(consensusesPr, intersectTab, "Repeat Type", 
              "/scratch/pdelangen/projet_these/oldBackup/temp_POL2_lenient/repeatBedType.bed")              
# %%
# Fantom eRNA
addAnnotation(consensusesPr, intersectTab, "Fantom 5 bidirectionnal enhancers", showCol=None,
              inputBed="/scratch/pdelangen/projet_these/data/annotation/F5.hg38.enhancers.bed")
# %%
# Fantom TSS
addAnnotation(consensusesPr, intersectTab, "Fantom 5 TSSs", showCol=None,
              inputBed="/scratch/pdelangen/projet_these/data/annotation/hg38_fair+new_CAGE_peaks_phase1and2.bed")
# %%
# Dnase meuleman
addAnnotation(consensusesPr, intersectTab, "DNase meuleman", 
              inputBed="/scratch/pdelangen/projet_these/data/annotation/dnaseMeuleman.bed")
# %%
intersectTab.to_csv(paths.tempDir + "intersectIntergPol2.tsv", sep="\t", index=None)


# %%
