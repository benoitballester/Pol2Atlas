# %%
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils, plot_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
import matplotlib.pyplot as plt

# %%
# First merge peaks and generate data matrix
merger = peakMerger(paths.genomeFile, outputPath=paths.outputDir)
merger.mergePeaks(paths.peaksFolder, inferCenter=params.inferCenter, 
                  minOverlap=params.minOverlap, fileFormat=params.fileFormat)
merger.writePeaks()
# %%
highVar = np.mean(merger.matrix.T, axis=0) < 0.51
orderRows = matrix_utils.threeStagesHC(merger.matrix.T[:, highVar], "dice")
# orderCols = matrix_utils.threeStagesHC(merger.matrix, "dice")
# %%
plot_utils.plotHC(merger.matrix, merger.labels, paths.annotationFile, rowOrder=orderRows, colOrder=orderCol)
plt.savefig(paths.outputDir + "pseudoHC.png")
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
merger.clusterize(transpose=False, restarts=10, annotationFile=paths.annotationFile,
                  reDo=True)

# %%
# Intersect enrichments
remap_catalog = pr.read_bed(paths.remapFile)
enrichments = overlap_utils.computeEnrichForLabels(remap_catalog, merger.consensuses, merger.clustered[0])
orderedP = np.argsort(enrichments[2].loc[3])
enrichments[2].loc[3][orderedP][:20]
del remap_catalog
# %%
# Repeats
repeats = pr.read_bed(paths.repeatFile)
enrichments = overlap_utils.computeEnrichForLabels(repeats, merger.consensuses, merger.clustered[0])
orderedP = np.argsort(enrichments[2].loc[3])
enrichments[2].loc[3][orderedP][:10]
del repeats
# %%

from lib.pyGREAT import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
# %%
goEnrich = enricher.findEnriched(merger.consensuses[6==merger.clustered[0]], merger.consensuses)

goClass = "biological_process"

hasEnrich = goEnrich[goClass][2].index[goEnrich[goClass][2] < 0.05]
subset = enricher.matrices[goClass].loc[hasEnrich]
clustered = matrix_utils.graphClustering(subset, "dice", k=20, r=0.4, restarts=10)
topK = 5
for c in np.arange(clustered.max()+1):
    inClust = hasEnrich[clustered == c]
    strongest = goEnrich[goClass][2][inClust].sort_values()[:topK]
    print("-"*20)
    print(strongest)

print(goEnrich[goClass][2][(goEnrich[goClass][2] < 0.05) & (goEnrich[goClass][1] > 2)].sort_values())
# %%
