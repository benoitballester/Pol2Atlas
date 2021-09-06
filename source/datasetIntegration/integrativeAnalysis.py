# %%
from lib.peakMerge import peakMerger
import numpy as np
from settings import params, paths

# %%
# First merge peaks and generate data matrix
merger = peakMerger(paths.genomeFile, outputPath=paths.outputDir)
merger.mergePeaks(paths.peaksFolder, inferCenter=params.inferCenter, 
                  minOverlap=params.minOverlap)
# %%
# Summary plots

# %%
# Intersect statistics plots

# %%
# UMAP and pseudo-HC

# %% 
# Clustering

# %%
# Intersect enrichments
