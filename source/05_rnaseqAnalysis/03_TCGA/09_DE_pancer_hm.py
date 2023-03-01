# %%
import os
import sys
from joblib.externals.loky import get_reusable_executor

sys.path.append("./")
sys.setrecursionlimit(10000)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lib.utils import matrix_utils, plot_utils
from scipy.cluster import hierarchy
from scipy.stats import mannwhitneyu, ttest_ind
from settings import params, paths
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import fdrcorrection

files = os.listdir(paths.outputDir + "rnaseq/TumorVsNormal/")
files = [f for f in files if f.startswith("TCGA-")]
# %%
diffNormal = []
cancers = []
for f in files:
    try:
        deltas = pd.read_csv(paths.outputDir + f"rnaseq/TumorVsNormal/{f}/allWithStats.bed", sep="\t")
        diffNormal.append((deltas["DeltaRes"].values))
        cancers.append(f)
        print(f)
    except:
        continue
diffNormal = pd.DataFrame(np.transpose(diffNormal), columns=cancers)
# %%
from sklearn.preprocessing import StandardScaler
order = matrix_utils.twoStagesHClinkage(np.clip(diffNormal,-1,1), metric="euclidean")
# %%
from skimage.transform import rescale, resize, downscale_local_mean
resized = resize(diffNormal.iloc[order[0]].values, (2000, diffNormal.shape[1]), anti_aliasing=True, order=1)
# %%
import seaborn as sns
plt.figure(dpi=500)
plt.imshow(resized, cmap=sns.color_palette("vlag", as_cmap=True), interpolation="nearest", vmin=-np.percentile(diffNormal,95), vmax=np.percentile(diffNormal,95))
plt.xticks(np.arange(len(cancers)), cancers, rotation=90)
plt.gca().set_aspect(len(cancers)/2000.0)
plt.title("Mean pearson residual Tumor - Mean pearson residual Normal\nIn each cancer")
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
# %%
