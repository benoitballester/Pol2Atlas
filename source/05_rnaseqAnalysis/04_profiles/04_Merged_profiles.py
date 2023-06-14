# %%
import sys
sys.path.append("./")
from settings import params, paths
import os
import numpy as np
import pandas as pd
readsPerPol2Bin = {}
readsPerPol2Bin["TCGA"] = np.load(paths.tempDir + "pooled_TCGA_10bp.npy")
readsPerPol2Bin["ENCODE"] = np.load(paths.tempDir + "pooled_ENCODE_10bp.npy")
readsPerPol2Bin["GTEx"] = np.load(paths.tempDir + "pooled_GTEX_10bp.npy")
# %%
# Normalize by maximum signal
hm = {}
for k in readsPerPol2Bin:
    hm[k] = readsPerPol2Bin[k]/np.maximum(np.max(readsPerPol2Bin[k], axis=1)[:, None],1e-10)
# %%
from lib.utils import matrix_utils
allPooled = np.concatenate([hm["ENCODE"],hm["TCGA"], hm["GTEx"]], axis=1)
rowOrder, _ = matrix_utils.twoStagesHClinkage(allPooled)
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,3, figsize=(4.5,8), dpi=300, sharex="all",
                        gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [1,12]})
for i, k in enumerate(hm):
    dsSize = min(20000, hm[k].shape[0])
    downSampledMatrixView = np.zeros((dsSize, hm[k].shape[1]))
    reordered = hm[k][rowOrder]
    for j in range(len(reordered)):
        r = reordered[j]
        idx = int(dsSize * j / len(hm[k]))
        downSampledMatrixView[idx] += r
    axs[1,i].imshow(downSampledMatrixView, interpolation="lanczos",
                  aspect=downSampledMatrixView.shape[1]/downSampledMatrixView.shape[0]*4.)
    axs[0,i].set_title(k)
    axs[0,i].plot(np.arange(hm[k].shape[1]),np.mean(hm[k], axis=0))
    axs[1,i].set_xticks(np.linspace(0,99,9), np.linspace(-500,500,9, dtype=int), rotation=90)
    if i == 0:
        axs[0,i].set_ylabel(f"Transcriptional \nsignal", fontsize=8)
        axs[1,i].set_ylabel(f"{len(hm[k])} RNAP2 bound regions", fontsize=8)  
    axs[0,i].tick_params(
            axis='both',
            which='both',
            top=False,
            left=False,
            labelleft=False)
    axs[1,i].tick_params(
            axis='both',
            which='both',
            top=False,
            left=False,
            labelleft=False)
fig.tight_layout()
fig.savefig(paths.outputDir + "rnaseq/profiles/all_same_order.pdf")
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,3, figsize=(4.5,8), dpi=300, sharex="all",
                        gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [1,12]})
for i, k in enumerate(hm):
    rowOrder, _ = matrix_utils.twoStagesHClinkage(hm[k])
    dsSize = min(20000, hm[k].shape[0])
    downSampledMatrixView = np.zeros((dsSize, hm[k].shape[1]))
    reordered = hm[k][rowOrder]
    for j in range(len(reordered)):
        r = reordered[j]
        idx = int(dsSize * j / len(hm[k]))
        downSampledMatrixView[idx] += r
    axs[1,i].imshow(downSampledMatrixView, interpolation="lanczos",
                  aspect=downSampledMatrixView.shape[1]/downSampledMatrixView.shape[0]*4.)
    axs[0,i].set_title(k)
    axs[0,i].plot(np.arange(hm[k].shape[1]),np.mean(hm[k], axis=0))
    axs[1,i].set_xticks(np.linspace(0,99,9), np.linspace(-500,500,9, dtype=int), rotation=90)
    if i == 0:
        axs[0,i].set_ylabel(f"Transcriptional \nsignal", fontsize=8)
        axs[1,i].set_ylabel(f"{len(hm[k])} RNAP2 bound regions", fontsize=8)  
    axs[0,i].tick_params(
            axis='both',
            which='both',
            top=False,
            left=False,
            labelleft=False)
    axs[1,i].tick_params(
            axis='both',
            which='both',
            top=False,
            left=False,
            labelleft=False)
fig.tight_layout()
fig.savefig(paths.outputDir + "rnaseq/profiles/all_independant_order.pdf")
# %%
fullPooled = readsPerPol2Bin["ENCODE"] + readsPerPol2Bin["TCGA"] + readsPerPol2Bin["GTEx"]
hm = fullPooled / np.maximum(np.max(fullPooled, axis=1)[:, None],1e-10)
# %%
from lib.utils import matrix_utils
from fastcluster import linkage_vector, linkage

# HC = linkage(hm, method="ward", metric="euclidean")
rowOrder, _ = matrix_utils.twoStagesHClinkage(hm)
hm = hm[rowOrder]

dsSize = min(20000, hm.shape[0])
downSampledMatrixView = np.zeros((dsSize, hm.shape[1]))
for i in range(len(hm)):
    r = hm[i]
    idx = int(dsSize * i / len(hm))
    downSampledMatrixView[idx] += r


import matplotlib.pyplot as plt
plt.figure(dpi=300)
plt.tick_params(
axis='both',
which='both',
top=False,
left=False,
labelleft=False)
plt.xticks(np.linspace(0,99,9), np.linspace(-500,500,9, dtype=int),fontsize=5)
plt.ylabel(f"Pol II consensuses", fontsize=5)
plt.xlabel(f"Distance to centroid (bp)", fontsize=5)
plt.imshow(downSampledMatrixView, interpolation="lanczos",
           aspect=downSampledMatrixView.shape[1]/downSampledMatrixView.shape[0]*40/20, 
           vmin=0, vmax=np.max(downSampledMatrixView))
plt.title("Read count (normalized by row maximum)", fontsize=10)
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/profiles/full_pooled.pdf", bbox_inches="tight")
plt.show()
plt.figure(dpi=300)
plt.plot(np.sum(hm, axis=0))
plt.xticks(np.linspace(0,99,9), np.linspace(-500,500,9, dtype=int))
plt.ylabel(f"Pooled transcriptional signal")
plt.xlabel(f"Distance to RNAP2 centroid (bp)")
plt.savefig(paths.outputDir + "rnaseq/profiles/profile_1kb_full_pooled.pdf", bbox_inches="tight")

# %%