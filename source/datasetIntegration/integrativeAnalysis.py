# %%
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils, plot_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
import matplotlib.pyplot as plt
# %%
'''
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils, plot_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
import matplotlib.pyplot as plt
import pickle
merger = pickle.load(open(paths.outputDir + "merger", "rb"))
'''
# %%
# First merge peaks and generate data matrix
merger = peakMerger(paths.genomeFile, outputPath=paths.outputDir)
merger.mergePeaks(paths.peaksFolder, inferCenter=params.inferCenter, 
                  minOverlap=params.minOverlap, fileFormat=params.fileFormat)
merger.writePeaks()

# %%
highVar = np.mean(merger.matrix.T, axis=0) < 0.51
orderRows = matrix_utils.threeStagesHC(merger.matrix.T[:, highVar], "dice")
orderCols = matrix_utils.threeStagesHC(merger.matrix, "dice")
# %%
annotationFile = pd.read_csv(paths.annotationFile, sep="\t", index_col="Sample")
labels = annotationFile.loc[merger.labels]["Annotation"]
plot_utils.plotHC(merger.matrix, labels, merger.matrix, annotationPalette=paths.polIIannotationPalette, rowOrder=orderRows, colOrder=orderCols)
plt.savefig(paths.outputDir + "pseudoHC.pdf")
# %%
# UMAP of samples
merger.umap(transpose=True, altMatrix=merger.matrix.T[:, highVar], annotationFile=paths.annotationFile, annotationPalette=paths.polIIannotationPalette, reDo=False)
# %%
# Clustering samples
merger.clusterize(transpose=True, restarts=100, annotationFile=paths.annotationFile, annotationPalette=paths.polIIannotationPalette)
# %%
# UMAP of consensus peaks
merger.umap(transpose=False, annotationFile=paths.annotationFile, reDo=False, annotationPalette=paths.polIIannotationPalette)
# %% 
# Clustering consensus peaks
merger.clusterize(transpose=False, restarts=10, annotationFile=paths.annotationFile,
                  reDo=False, annotationPalette=paths.polIIannotationPalette)

# %%
# Intersect enrichments
try:
    os.mkdir(paths.outputDir + "cluster_enrichments/")
except FileExistsError:
    pass
# %%
# Remap
overlap_utils.computeEnrichForLabels(pr.read_bed(paths.remapFile), 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/remap")
# %%
# Repeats
overlap_utils.computeEnrichForLabels(pr.read_bed(paths.repeatFile), 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/repeats")
# %%                                
# DNase meuleman
dnase = pr.read_bed("/scratch/pdelangen/projet_these/data/annotation/dnaseMeuleman.bed")
fc, q, p = overlap_utils.computeEnrichForLabels(dnase, 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/dnaseIndex")
# %%
import seaborn as sns
hmOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)
plt.figure(figsize=(6,4), dpi=500)
sns.heatmap(np.clip(-np.log(p.values[:, hmOrder]),0,300), cmap="viridis", linewidths=0.1, linecolor='black')
plt.gca().set_aspect(p.shape[1]/p.shape[0])
plt.xlabel(f"{p.shape[1]} Pol II clusters")
plt.ylabel("Dnase label")
plt.yticks(np.arange(len(p))+0.5, p.index, rotation=0)
plt.xticks([],[])
plt.title("-log10(Intersection enrichment p-value)")
plt.savefig(f"{paths.outputDir}/cluster_enrichments/dnaseHeatmap.pdf", bbox_inches="tight")
plt.show()
# %%
# GO terms
from lib.pyGREATglm import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
# %%
# testReg = pd.read_csv(paths.tempDir + "globallyProg.bed", sep="\t", header=None)
for i in range(np.max(merger.clustered[0])+1):
    testReg = merger.consensuses[merger.clustered[0]==i]
    goEnrich = enricher.findEnriched(testReg, merger.consensuses)
    enricher.plotEnrichs(enrichDF)
# %%
# %%
