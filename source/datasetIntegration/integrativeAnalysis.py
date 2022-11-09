# %%
import sys
sys.path.append("./")
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils, plot_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.setrecursionlimit(100000)
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
# GO terms
from lib.pyGREATglm import pyGREAT
enricher = pyGREAT(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
if params.removeGeneTail5kb:
    gencode = enricher.txList
    consensusesPr = merger.consensuses[[0,6,7]]
    consensusesPr["Name"] = np.arange(len(consensusesPr))
    consensusesPr.columns = ["Chromosome", "Start", "End", "Name"]
    consensusesPr = pr.PyRanges(consensusesPr)
    tes_ext = 5000
    tails = gencode.copy()
    tesP = tails.loc[tails["Strand"] == "+"]["End"]
    tails.loc[tails["Strand"] == "+", ["Start", "End"]] = np.array([tesP, tesP+tes_ext]).T
    tesM = tails.loc[tails["Strand"] == "-"]["Start"]
    tails.loc[tails["Strand"] == "-", ["Start", "End"]] = np.array([tesM-tes_ext, tesM]).T
    tailedPol2 = consensusesPr.join(pr.PyRanges(tails), False, apply_strand_suffix=False).as_df()
    tailedPol2.drop(["Start_b", "End_b", "Strand"], axis=1, inplace=True)
    toRemove = tailedPol2["Name"].values
    merger.consensuses = merger.consensuses.drop(toRemove)
    merger.matrix = np.delete(merger.matrix, toRemove, axis=0)
# %%
highVar = np.mean(merger.matrix.T, axis=0) < 0.51
orderRows = matrix_utils.threeStagesHC(merger.matrix.T[:, highVar], "yule")
orderCols = matrix_utils.threeStagesHC(merger.matrix, "dice")
# %%
annotationFile = pd.read_csv(paths.annotationFile, sep="\t", index_col="Sample")
labels = annotationFile.loc[merger.labels]["Annotation"]
plot_utils.plotHC(merger.matrix, labels, merger.matrix,
                  annotationPalette=paths.polIIannotationPalette, 
                  rowOrder=orderRows, colOrder=orderCols,
                  rescale=False)
plt.savefig(paths.outputDir + "pseudoHC.pdf")
# %%
# UMAP of samples
merger.umap(transpose=True, metric="yule", altMatrix=merger.matrix.T[:, highVar], 
            annotationFile=paths.annotationFile, annotationPalette=paths.polIIannotationPalette, 
            reDo=True)
# %%
# Clustering samples
merger.clusterize(transpose=True, r=0.5, metric="yule", restarts=100, annotationFile=paths.annotationFile, annotationPalette=paths.polIIannotationPalette)
# %%
# UMAP of consensus peaks
merger.umap(transpose=False, metric="dice", altMatrix=merger.matrix, annotationFile=paths.annotationFile, reDo=True, annotationPalette=paths.polIIannotationPalette)
# %% 
# Clustering consensus peaks
merger.clusterize(transpose=False, r=0.5, metric="dice", restarts=10, annotationFile=paths.annotationFile,
                  reDo=False, annotationPalette=paths.polIIannotationPalette)

# %%
# Intersect enrichments
try:
    os.mkdir(paths.outputDir + "cluster_enrichments/")
except FileExistsError:
    pass
# %%
# Annotation Specific
enrichs = merger.topPeaksPerAnnot(paths.annotationFile)
# %%
# Remap
overlap_utils.computeEnrichForLabels(pr.read_bed(paths.remapFile), 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/remap")
# %%
# Repeats
overlap_utils.computeEnrichForLabels(pr.read_bed(paths.repeatTypeBed), 
                                    merger.consensuses, merger.clustered[0], 
                                    paths.outputDir + "cluster_enrichments/repeats")
# %%                                
# DNase meuleman
dnase = pr.read_bed(paths.dnaseMeuleman)
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
# testReg = pd.read_csv(paths.tempDir + "globallyProg.bed", sep="\t", header=None)
for i in range(np.max(merger.clustered[0])+1):
    testReg = merger.consensuses[merger.clustered[0]==i][[0,1,2]]
    testReg.columns = ["Chromosome", "Start", "End"]
    testReg = pr.PyRanges(testReg)
    goEnrich = enricher.findEnriched(testReg, pr.read_bed(paths.outputDir+"consensuses.bed"))
    goEnrich.to_csv(paths.outputDir + f"cluster_enrichments/go_enrich_{i}.csv")
    enricher.plotEnrichs(goEnrich, savePath=paths.outputDir + f"cluster_enrichments/GO_fc_{i}.pdf")
    if (goEnrich["BH corrected p-value"] < 0.05).sum() >= 2:
        enricher.clusterTreemap(goEnrich, output=paths.outputDir + f"cluster_enrichments/GO_treemap_{i}.pdf")
# %%
