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
# UMAP of samples
merger.umap(transpose=True, annotationFile=paths.annotationFile)
# %%
# Clustering samples
merger.clusterize(transpose=True, restarts=100, annotationFile=paths.annotationFile)
# %%
# UMAP of consensus peaks
merger.umap(transpose=False, annotationFile=paths.annotationFile)
# %% 
# Clustering consensus peaks
merger.clusterize(transpose=False, restarts=1, annotationFile=paths.annotationFile,
                  reDo=True)
# %%
# Intersect enrichments
