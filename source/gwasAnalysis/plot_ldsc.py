# %%
import pandas as pd
import numpy as np
import sys
import os
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
enrichDF = pd.DataFrame()
pvalDF = pd.DataFrame()
for k in results:
    enrichDF[k] = results[k]["Enrichment"]
    pvalDF[k] = results[k]["Enrichment_p"]
# %%
from statsmodels.stats.multitest import fdrcorrection
qvalDf = pvalDF.copy()*np.prod(pvalDF.shape)
# %%
from lib.utils import plot_utils
import matplotlib.pyplot as plt
for clust in enrichDF.index:
    print(clust)
    enrichDF.loc[clust][(enrichDF.loc[clust] > 1.0) & (qvalDf.loc[clust] < 0.05)].sort_values()[::-1]
    fig, ax = plt.subplots(figsize=(2,2), dpi=500)
    plot_utils.enrichBarplot(ax, enrichDF.loc[clust], qvalDf.loc[clust], fcMin=2.0, order_by="fc")
    ax.set_xlabel("Heritability enrichment")
    fig.savefig(f"{paths.outputDir}/cluster_enrichments/ldsc_{clust}.png", bbox_inches="tight")
    # plt.show()
    plt.close()
# %%
