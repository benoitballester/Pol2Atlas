# %%
# Script to be used with our featureCount script
import numpy as np
import pandas as pd
from settings import params, paths

# %%
# Scan all count files
allFiles = os.listdir(paths.countDirectory)
allFiles = [f for f in allFiles if f.endswith("txt.gz")]
allCounts = []
keys = []
scalingFactors = []
tabsScaling = []
# Read all files and append columns to the count matrix
for f in allFiles:
    key = f[:-7]
    keys.append(key)
    counts = pd.read_csv(paths.countDirectory + f, skiprows=1)
    totalReads = pd.read_csv(paths.countDirectory + f"{key}.counts.summary",
                            sep="\t", index_col=0).loc[["Assigned", "Unassigned_NoFeatures"]].sum().values[0]
    scalingFactors.append(totalReads)   # Normalize by total number of reads
    tabsScaling.append(pd.read_csv(paths.countDirectory + f"{key}.counts.summary",
                            sep="\t", index_col=0))
    allCounts.append(counts.values)
allCounts = np.concatenate(allCounts, axis=1)
# Remove consensuses full of zeros
nzPos = np.sum(allCounts, axis=1) > 0.0016
allCounts = allCounts[nzPos]
allCounts = allCounts.T
# Adjusts indexes to skipped zero count consensuses
coordinates = pd.read_csv(paths.atlasPath, sep="\t", index_col=0)[nzPos]

# %%
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
try:
    os.mkdir(paths.outputDir + "rnaseq/")
except:
    pass
# Write the matrix and the consensuses
mmwrite(paths.outputDir + "rnaseq/" + "counts.mtx", csr_matrix(allCounts))
coordinates.to_csv(paths.outputDir + "rnaseq/consensuses.bed", sep="\t")
# Save file names (used for accessing the tcga annotation)
pd.DataFrame(keys).to_csv(paths.outputDir + "rnaseq/keys.txt", sep="\t", index=False, header=None)

# %%
