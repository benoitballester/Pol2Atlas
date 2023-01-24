# %%
import pandas as pd
import numpy as np
import sys
import os
from statsmodels.stats.multitest import fdrcorrection
sys.path.append("./")
from settings import params, paths

lookupTable = pd.read_csv(f"{paths.ldscFilesPath}/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t")[["phenotype", "description"]]
lookupTable.set_index("phenotype", inplace=True)
lookupTable.index = lookupTable.index.astype(str)

allFiles = os.listdir(paths.outputDir + "ldsc/")
allFiles = [f for f in allFiles if f.endswith(".results")]
results = dict()
for f in allFiles:
    df = pd.read_csv(paths.outputDir + f"ldsc/{f}", sep="\t")
    df["Category"] = df["Category"].str.split("/cluster_",2,True)[1].str.split("L2_0",2,True)[0].astype("str")
    df.set_index("Category", inplace=True)
    cats = df.index
    trait = f.split(".")[0]
    results[np.unique(lookupTable.loc[trait]["description"])[0]] = df
# %%
sigEnrichs = pd.DataFrame()
hmOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)[::-1]
for k in results:
    sigEnrichs[k] = -np.log10(results[k]["Enrichment_p"]) * (results[k]["Enrichment"] > 1.0).astype(float)
# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
normed = (sigEnrichs)/np.mean(sigEnrichs.values, axis=1)[:, None]
ax=sns.clustermap(sigEnrichs, row_cluster=False, col_cluster=True, metric="correlation",
                xticklabels=True, yticklabels=True, rasterized=True)
ax.ax_heatmap.axes.set_xticklabels(ax.ax_heatmap.axes.get_xticklabels(), fontsize=2)
plt.savefig(paths.tempDir + "test_ldsc.pdf")
# %%
