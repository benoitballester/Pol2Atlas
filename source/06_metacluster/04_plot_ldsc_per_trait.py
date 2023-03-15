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

allFiles = os.listdir(paths.outputDir + "ldsc_meta/")
allFiles = [f for f in allFiles if f.endswith(".results")]
results = dict()
for f in allFiles:
    df = pd.read_csv(paths.outputDir + f"ldsc_meta/{f}", sep="\t")
    df["Category"] = df["Category"].str.split("/liftedClusters/",2,True)[1].str.split("L2_0",2,True)[0].astype("str")
    df.set_index("Category", inplace=True)
    cats = df.index
    trait = f.split(".")[0]
    results[np.unique(lookupTable.loc[trait]["description"])[0]] = df
# %%
sigEnrichs = pd.DataFrame()
hmOrder = np.loadtxt(paths.outputDir + "clusterBarplotOrder.txt").astype(int)[::-1]
for k in results:
    sigEnrichs[k] = results[k]["Enrichment"] * (results[k]["Enrichment_p"] < 0.01/7463).astype(float)
sigEnrichs = np.maximum(0,sigEnrichs)
# %%
# %%
# Plot top markers set for each gwas
import seaborn as sns
import matplotlib.pyplot as plt
usedTraits = ["Birth weight of first child", 
              "Ever had prolonged feelings of sadness or depression",
              "Non-cancer illness code, self-reported: high cholesterol", 
              "Non-cancer illness code, self-reported: hypothyroidism/myxoedema"]
palette = pd.read_csv(paths.polIIannotationPalette, sep=",", index_col=0)
for trait in usedTraits:
    reordered = sigEnrichs[trait].sort_values()[::-1][:5]
    idx = [i.split("halfDatasets_")[1].replace("_", " ").replace("-", "/") for i in reordered.index]
    col = palette.loc[idx]
    plt.figure(dpi=500)
    plt.barh(np.arange(len(reordered))[::-1], reordered.values, color=col.values)
    plt.yticks(np.arange(len(reordered))[::-1], idx)
    plt.title(trait)
    plt.xlabel("Heritability enrichment")
    plt.ylabel("Markers")
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    for i, vals in enumerate(reordered.values):
        if vals == 0.0:
            plt.text(plt.xlim()[1]*0.01, 4-i, "n.s", va="center")
    name = trait.replace("/", "-")
    plt.savefig(paths.outputDir + f"ldsc_meta_figures/{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
# %%
with open(paths.tempDir + "end0604.txt", "w") as f:
    f.write("1")