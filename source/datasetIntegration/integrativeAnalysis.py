# %%
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
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
remap_catalog = pr.read_bed(paths.remapFile)
allPolII = overlap_utils.dfToPrWorkaround(merger.consensuses[[0,1,2]])
enrichments = overlap_utils.computeEnrichForLabels(remap_catalog, allPolII, merger.clustered[0])
orderedP = np.argsort(enrichments[2].loc[19])
enrichments[2].loc[19][orderedP][:20]
del remap_catalog
# %%
# Repeats
repeats = pr.read_bed(paths.repeatFile)
enrichments = overlap_utils.computeEnrichForLabels(repeats, allPolII, merger.clustered[0])
orderedP = np.argsort(enrichments[2].loc[3])
enrichments[2].loc[3][orderedP][:10]
del repeats
# %%
