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
plot_utils.plotHC(merger.matrix, merger.labels, paths.annotationFile, rowOrder=orderRows, colOrder=orderCols)
plt.savefig(paths.outputDir + "pseudoHC.pdf")
# %%
# UMAP of samples
merger.umap(transpose=True, altMatrix=merger.matrix.T[:, highVar], annotationFile=paths.annotationFile, reDo=True)
# %%
# Clustering samples
merger.clusterize(transpose=True, restarts=100, annotationFile=paths.annotationFile)
# %%
# UMAP of consensus peaks
merger.umap(transpose=False, annotationFile=paths.annotationFile, reDo=False)
# %% 
# Clustering consensus peaks
merger.clusterize(transpose=False, restarts=10, annotationFile=paths.annotationFile,
                  reDo=False)

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
overlap_utils.computeEnrichForLabels(pr.read_bed("/scratch/pdelangen/projet_these/data/annotation/dnaseMeuleman.bed"), 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/dnaseIndex")
# %%
# GO terms
from lib.pyGREAT import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
# %%
# testReg = pd.read_csv(paths.tempDir + "globallyProg.bed", sep="\t", header=None)
for i in range(np.max(merger.clustered[0])+1):
    i = 9
    testReg = merger.consensuses[merger.clustered[0]==i]
    goEnrich = enricher.findEnriched(testReg, merger.consensuses)
    goEnrich.set_index("name", inplace=True)
    fig, ax = plt.subplots(figsize=(2,2),dpi=500)
    plot_utils.enrichBarplot(ax, goEnrich["p_value"], goEnrich["p_value"], fcMin=0.0, order_by="qval")
    break
    fig.savefig(f"{paths.outputDir}/cluster_enrichments/GO_fc_{i}.png", bbox_inches="tight")
    plt.show()
    plt.close()
# %%
# %%
