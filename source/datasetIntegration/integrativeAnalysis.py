# %%
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils
import numpy as np
from settings import params, paths

# %%
# First merge peaks and generate data matrix
merger = peakMerger(paths.genomeFile, outputPath=paths.outputDir)
merger.mergePeaks(paths.peaksFolder, inferCenter=params.inferCenter, 
                  minOverlap=params.minOverlap, fileFormat=params.fileFormat)
merger.writePeaks()
# %% 
# Assign genomic context

# %%
# Summary plots

# %%
# Intersect statistics plots

# %%
# UMAP and pseudo-HC
merger.umap(transpose=True, annotationFile=paths.annotationFile)
merger.umap(transpose=False, annotationFile=paths.annotationFile)
# %% 
# Clustering
merger.clusterize(transpose=True, restarts=100, annotationFile=paths.annotationFile)
# %%
# Intersect enrichments
