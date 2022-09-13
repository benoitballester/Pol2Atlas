import numpy as np
import pandas as pd
import subprocess
from settings import params, paths
from statsmodels.stats.multitest import fdrcorrection

# %%
consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
def runScript(script, argumentList, outFile=None):
    # Runs the command as a standard bash command
    # script is command name without path
    # argumentList is the list of argument that will be passed to the command
    if outFile == None:
        subprocess.run([script] + argumentList)
    else:
        with open(outFile, "wb") as outdir:
            subprocess.run([script] + argumentList, stdout=outdir)

datasets = ["TCGA", "ENCODE", "GTEX"]

prefix = paths.tempDir + "ENCODE_"
tempCountPath = prefix + "CountsDE.mtx"
tempNames = prefix + "NzIdxDE.txt"
tempSf = prefix + "SfDE.txt"
tempLabels = prefix + "LabelsDE.txt"
runScript("Rscript", ["--vanilla", "lib/limma.r", tempCountPath, tempLabels, tempNames,
                     tempSf, deFolder])

# %%
pctThreshold = 1/3.0
lfcMin = 0.25
for i in np.unique(ann):
    print(eq[i])
    labels = ann == i
    name = eq[i].replace(" ", "_")
    # labels = np.random.permutation(labels)
    res2 = pd.read_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/ groups{name} .csv", index_col=0)
    res2 = res2.iloc[np.argsort(res2.index)]
    res2["P.Value"] = np.where(res2["logFC"].values > 0, res2["P.Value"]/2, 1-res2["P.Value"]/2)
    res2["adj.P.Val"] = fdrcorrection(res2["P.Value"])[1]
    sig = res2["adj.P.Val"] < 0.05
    minpct = np.mean(counts[labels] > 0.5, axis=0) > max(pctThreshold, 1.5/labels.sum())
    lfc = res2["logFC"] > lfcMin
    print(sig.sum())
    sig = sig & minpct & lfc
    print(sig.sum())
    res = pd.DataFrame(res2[["adj.P.Val", "t"]], columns=consensuses.index[nzCounts], index=["pval", "stat"]).T
    res["Upreg"] = 1-sig.astype(int)
    res["lfc"] = -res2["logFC"]
    order = np.lexsort(res[["lfc","pval","Upreg"]].values.T)
    res["lfc"] = res2["logFC"]
    res["Upreg"] = sig.astype(int)
    res = res.iloc[order]
    res.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/res_{eq[i]}.csv")
    test = consensuses.iloc[sig.index[sig]]
    test.to_csv(paths.outputDir + f"rnaseq/encode_rnaseq/DE/bed_{eq[i]}", header=None, sep="\t", index=None)
    if len(test) == 0:
        continue
    '''
    pvals = enricherglm.findEnriched(test, background=consensuses)
    enricherglm.plotEnrichs(pvals)
    enricherglm.clusterTreemap(pvals, score="-log10(pval)", 
                                output=paths.outputDir + f"rnaseq/encode_rnaseq/DE/great_{eq[i]}.pdf")
    '''