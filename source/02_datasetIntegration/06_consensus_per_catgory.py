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
import pickle
merger = pickle.load(open(paths.outputDir + "merger", "rb"))
# %%
annot = pd.read_csv(paths.annotationFile, sep="\t", index_col="Sample").loc[merger.labels]
# %%
numDetected = pd.DataFrame(np.zeros((len(annot["Category"].unique()), len(merger.matrix)), dtype=int),
                           index=annot["Category"].unique(),)
for cat in annot["Category"].unique():
    numDetected.loc[cat] = np.sum(merger.matrix.T[annot["Category"].values==cat], axis=0)
# %%
detected_sets = {}
for cat in annot["Category"].unique():
    detected_sets[cat] = set(np.nonzero(numDetected.loc[cat].values)[0])
# %%
# Issues with pip for whatever reason, we can just copy paste the git and use it
from lib.matplotlib_venn import venn3
plt.figure(dpi=500)
venn_diagram = venn3([detected_sets[k] for k in detected_sets.keys()], 
                     set_labels=tuple(detected_sets.keys()), alpha=1.0)
plt.savefig(paths.outputDir + "descriptivePlots/venn_sample_category.pdf")
plt.show()
# %%
