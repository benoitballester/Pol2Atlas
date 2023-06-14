# %%
import sys
sys.path.append("./")
from settings import params, paths
import os
import numpy as np
import pandas as pd
import pyranges as pr

funcAnn = pr.read_bed(paths.tempDir + "consensuses_func_annot.tsv", as_df=True)
# %%
clusters = pd.read_csv(paths.outputDir + "clusterConsensuses_Labels.txt", header=None).values.ravel()
cluster_labels = pd.read_csv(paths.dataFolder + "dataset_annotation/cluster_annotation.csv",
                             sep="\t", index_col="Cluster")
# %%
consensus_label = cluster_labels.loc[clusters]
consensus_label["?"] = ""
consensus_label["?"][consensus_label["Annotation confidence"] == "Medium"] = "?"
consensus_label["?"][consensus_label["Annotation confidence"] == "Low"] = "??"
# %%
txt = [(funcAnn.iloc[i, 3] + ";" + consensus_label.iloc[i, 0] + consensus_label.iloc[i, 2]).replace(" ", "_")  for i in range(len(funcAnn))]
# %%
funcAnn["Name"] = txt
# %%
prom = np.array(["Promoter-like" in l for l in txt])
enh = np.array(["Enhancer-like" in l for l in txt])
lnc = np.array(["LNC-like" in l for l in txt])
reg = np.array(["Regulatory" in l for l in txt])
colors = np.array([[166,166,166]]*len(txt))
colors[reg] = [191,0,255]
colors[enh] = [255,0,0]
colors[lnc] = [0,153,0]
colors[prom] = [255,140,26]
colors = [str(tuple(c))[1:-1].replace(" ","") for c in colors]
funcAnn["rgb"] = colors
# %%
funcAnn.to_csv(paths.outputDir + "annotated_consensuses.bed", header=None, sep="\t", index=None)
# %%
