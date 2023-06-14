# %%
import sys
sys.path.append("./")
from settings import params, paths
import os
import numpy as np
import pandas as pd

try:
    os.mkdir(paths.outputDir + "rnaseq/profiles/")
except:
    pass
try:
    os.mkdir(paths.outputDir + "rnaseq/profiles/encode/")
except:
    pass


directory = "/shared/projects/pol2_chipseq/rnaseq_counts_10bp/encode_10bp/"
files = os.listdir(directory)
# files = ["fffaeee5-ad28-4ad0-87f2-4a3751976710.counts.txt"]
readsPerPol2Bin = None
avgScale = 0
for f in files:
    try:
        countDF = pd.read_csv(directory+f, skiprows=1, sep="\t", dtype="int32").values.reshape(-1,100)
        if readsPerPol2Bin is None:
            readsPerPol2Bin = np.zeros_like(countDF, dtype=float)
        readsPerPol2Bin += countDF / np.mean(countDF)
        avgScale += np.mean(countDF)
    except pd.errors.EmptyDataError:
        pass
readsPerPol2Bin /= avgScale
np.save(paths.tempDir + "pooled_ENCODE_10bp.npy", readsPerPol2Bin)
# %%
readsPerPol2Bin = np.load(paths.tempDir + "pooled_ENCODE_10bp.npy")
# %%
hasreads = readsPerPol2Bin.max(axis=1) > 0
print(hasreads.sum())

# Normalize by maximum signal
hm = readsPerPol2Bin[hasreads]/np.max(readsPerPol2Bin[hasreads], axis=1)[:, None]
selected = np.random.permutation(len(hm))[:20000]
hm = hm[selected]
# %%
import scipy.cluster.hierarchy as hierarchy
from fastcluster import linkage_vector, linkage

HC = linkage(hm, method="ward", metric="euclidean")
rowOrder = hierarchy.leaves_list(HC)
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
plt.ylabel(f"{hasreads.sum()} Pol II consensuses (with > 10 reads)", fontsize=5)
plt.xlabel(f"Distance to centroid (bp)", fontsize=5)
plt.imshow(downSampledMatrixView, interpolation="lanczos",
           aspect=downSampledMatrixView.shape[1]/downSampledMatrixView.shape[0]*40/20, 
           vmin=0, vmax=np.max(downSampledMatrixView))
plt.title("Pooled normalized read count (normalized by row maximum)", fontsize=10)
plt.tight_layout()
plt.savefig(paths.outputDir + "rnaseq/profiles/encode/profile_1kb_heatmap.pdf", bbox_inches="tight")
plt.show()
# %%
plt.figure(dpi=300)
plt.plot(np.sum(readsPerPol2Bin[hasreads]/np.max(readsPerPol2Bin[hasreads], axis=1)[:, None], axis=0))
plt.xticks(np.linspace(0,99,9), np.linspace(-500,500,9, dtype=int))
plt.ylabel(f"Pooled transcriptional signal")
plt.xlabel(f"Distance to RNAP2 centroid (bp)")
plt.savefig(paths.outputDir + "rnaseq/profiles/encode/profile_1kb.pdf", bbox_inches="tight")

# %%
