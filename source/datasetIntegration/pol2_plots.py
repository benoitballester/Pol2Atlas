# %%
# Pol II only, Run this after the integrative analysis
from lib.peakMerge import peakMerger
from lib.utils import overlap_utils, matrix_utils, plot_utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

donutSize = 0.3
target = "POLR2A"
listFiles = os.listdir(paths.peaksFolder)
outputDir = paths.outputDir + "descriptivePlots/"
try:
    os.mkdir(outputDir)
except FileExistsError:
    pass
# %%
isEncode = [1 if i.startswith("ENCS") else 0 for i in listFiles]
counts = np.bincount(isEncode)
labels = ["GEO", "ENCODE"]
palette = plot_utils.getPalette(isEncode)[0]
plot_utils.donutPlot(donutSize, counts, counts.sum(), labels, 
          counts.sum(), target + " ChIP-seq", palette, fName=outputDir + "expOrigin", showPct=False)

# %%
annotation = pd.read_csv(paths.annotationFile, sep="\t", index_col="Sample")
fileAnnot, annots = pd.factorize(annotation.loc[listFiles]["Annotation"])
counts = np.bincount(fileAnnot)
palette = plot_utils.getPalette(fileAnnot)[0]
plot_utils.donutPlot(donutSize, counts, counts.sum(), annots, 
          counts.sum(), target + " ChIP-seq", palette, fName=outputDir + "expBiotype", showPct=False)
# %%
plt.figure(dpi=300, figsize=(3,2))
distribPeaks = np.sum(merger.matrix, 1)
maxVal = int(np.percentile(distribPeaks,95))
n, bins, _ = plt.hist(np.clip(distribPeaks, 0, maxVal), np.arange(0, 1+maxVal))
plt.text(bins[-2]*0.5+bins[-1]*0.5, n[-1] + 0.05*plt.ylim()[1], 
         str(maxVal) + "+", ha="center", size=8)
plt.xlabel("Peak observed in x experiments", size=8)
plt.ylabel("Number of peaks", size=8)
plt.tick_params(labelsize=8)
plt.savefig(paths.outputDir + "descriptivePlots/distrib_overlap_per_peak.pdf", bbox_inches="tight")
plt.show()