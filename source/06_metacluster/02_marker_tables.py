# %%
import os
import sys
sys.path.append("./")
sys.setrecursionlimit(10000)
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from settings import params, paths
from scipy.sparse import csr_matrix

figPath = paths.outputDir + "markerTables/"
try:
    os.mkdir(figPath)
except FileExistsError:
    pass
indices = dict()
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[[0,1,2,3]]
consensuses.columns = ["Chromosome", "Start", "End", "ID"]
# %%
# Marker per dataset per annot table
filesEncode = os.listdir(paths.outputDir + "rnaseq/encode_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "Encode_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/encode_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) < 1:
        continue
    indices[name] = vals
filesPol2 = os.listdir(paths.outputDir + "enrichedPerAnnot/")
filesPol2 = [f for f in filesPol2 if not f.endswith("_qvals.bed")]
for f in filesPol2:
    name = "Pol2_" + f[:-4]
    try:
        bed = pd.read_csv(paths.outputDir + "enrichedPerAnnot/" + f, header=None, sep="\t")
        if len(bed) < 1:
            continue
        indices[name] = bed[3].values
    except pd.errors.EmptyDataError:
        continue
# Files GTex
filesEncode = os.listdir(paths.outputDir + "rnaseq/gtex_rnaseq/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "GTex_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/gtex_rnaseq/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals


# Files TCGA
filesEncode = os.listdir(paths.outputDir + "rnaseq/TCGA2/DE/")
filesEncode = [f for f in filesEncode if f.startswith("res_")]
for f in filesEncode:
    name = "TCGA_" + f[4:-4]
    res = pd.read_csv(paths.outputDir + "rnaseq/TCGA2/DE/" + f, index_col=0)["Upreg"] 
    vals = res.index[res==1].values
    if len(vals) == 0:
        continue
    indices[name] = vals

rows = []
cols = []
data = []
for i, k in enumerate(indices.keys()):
    idx = list(indices[k])
    cols += idx
    rows += [i]*len(idx)
    data += [True]*len(idx)

# Build matrix    
mat = csr_matrix((data, (rows, cols)), shape=(len(indices), len(consensuses)), dtype=bool)
mat = pd.DataFrame(mat.todense(), index=indices.keys()).astype(int).T
markerTable = pd.concat([consensuses, mat], axis=1)
markerTable.to_csv(figPath + "markerPerDatasetPerTissue.csv", sep="\t", index=None)
# Text format
txtTissueMarkers = [";".join([f for m, f in zip(mat.values[i], mat.columns) if m==1]) for i in range(len(consensuses))]
# %%
# Markers survival per cancer
survMarkersPath = paths.outputDir + "rnaseq/Survival/"
folders = [f for f in os.listdir(survMarkersPath) if f.startswith("TCGA-")]
markerVectors = []
for f in folders:
    try:
        prog = pd.read_csv(survMarkersPath + f + "/prognostic.bed", sep="\t", header=None)
        vec = np.zeros(len(consensuses), dtype="int8")
        vec[prog[3]] = 1
        markerVectors.append(vec)
    except:
        markerVectors.append(np.zeros(len(consensuses), dtype="int8"))
# Text format
arr = np.array(markerVectors)
txtSurv = [";".join([f for m, f in zip(arr[:,i],folders) if m==1]) for i in range(len(consensuses))]
# Add "pancancer" column
pancancer = pd.read_csv(survMarkersPath + "globally_prognostic.bed", sep="\t", header=None)
vecPancancerSurv = np.zeros(len(consensuses), dtype="int8")
vecPancancerSurv[pancancer[3]] = 1
markerVectors.append(vecPancancerSurv)
folders.append("Pan-cancer")
markerVectors = pd.DataFrame(markerVectors, index=folders).T
markerTable = pd.concat([consensuses, markerVectors], axis=1)
markerTable.to_csv(figPath + "survivalMarkerPerCancer.csv", sep="\t", index=None)
# %%
# Markers DE per cancer
deMarkersPath = paths.outputDir + "rnaseq/TumorVsNormal/"
folders = [f for f in os.listdir(deMarkersPath) if f.startswith("TCGA-")]
markerVectors = []
for f in folders:
    try:
        prog = pd.read_csv(deMarkersPath + f + "/allDE.bed", sep="\t", header=None)
        vec = np.zeros(len(consensuses), dtype="int8")
        vec[prog[3]] = 1
        markerVectors.append(vec)
    except:
        markerVectors.append(np.zeros(len(consensuses), dtype="int8"))
# Text format
arr = np.array(markerVectors)
txtDE = [";".join([f for m, f in zip(arr[:,i],folders) if m==1]) for i in range(len(consensuses))]
# Add "pancancer" column
pancancer = pd.read_csv(deMarkersPath + "globally_DE.bed", sep="\t", header=None)
vecPancancerDE = np.zeros(len(consensuses), dtype="int8")
vecPancancerDE[pancancer[3]] = 1
markerVectors.append(vecPancancerDE)
folders.append("Pan-cancer")
markerVectors = pd.DataFrame(markerVectors, index=folders).T
markerTable = pd.concat([consensuses, markerVectors], axis=1)
markerTable.to_csv(figPath + "DEMarkerPerCancer.csv", sep="\t", index=None)

# %%
# Save as txt instead of binary
intersectTable = pd.read_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t")
intersectTable["Survival_marker"] = txtSurv
intersectTable["Pan-cancer_Survival_marker"] = vecPancancerSurv
intersectTable["DE_marker"] = txtDE
intersectTable["Pan-cancer_DE_marker"] = vecPancancerDE
intersectTable["Tissue_marker"] = txtTissueMarkers
intersectTable.to_csv(paths.outputDir + "allMarkers_allIntersects.csv", sep="\t", index=None)
# %%
