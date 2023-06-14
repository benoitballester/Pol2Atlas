# %%
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from settings import params, paths
# %%
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7013921/
tab = pd.read_csv(paths.hallmarksGenes, 
                  sep="\t", encoding = "ISO-8859-1")
# %%
with open(paths.tempDir + "hallmark_cancer_chg.gmt", "w") as f:
    for h in tab.columns:
        genes = tab[h].dropna().str.upper()
        genes = np.unique(genes)
        txt = "\t".join(genes)
        txt = h + "\t" + h + "\t" + txt + "\n"
        f.write(txt)
# %%
from lib.pyGREATglm import pyGREAT
enricher = pyGREAT(paths.tempDir + "hallmark_cancer_chg.gmt",
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile,
                          validGenes="all")
# %%
import pyranges as pr
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
globallyDE = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal/globally_DE.bed")
pvals = enricher.findEnriched(globallyDE, background=consensuses, minGenes=1, maxGenes=1e6)
pvals.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/pancancer_hallmarkGenes_enrichs.csv")
# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
res = pvals
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765) if p < 0.05 else (0.5,0.5,0.5) for p in res["P(Beta > 0)"].values]
sns.barplot(x=res.index, y=-np.log10(res["P(Beta > 0)"].values), palette=colors)
plt.xlabel("Cancer hallmark")
plt.xlim(plt.xlim()[0], plt.xlim()[1])
plt.xticks(rotation=90, fontsize=10)
plt.hlines(-np.log10(0.05), plt.xlim()[0], plt.xlim()[1], color="red", linestyle="dashed")
plt.ylabel("-log10(P-value)")
plt.gca().set_aspect(np.ptp(plt.xlim())/np.ptp(plt.ylim())*0.5)
plt.savefig(paths.outputDir + "rnaseq/TumorVsNormal/pancancer_hallmarkGenes_enrichs.pdf",
            bbox_inches="tight")
plt.show()
# %%
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
globallyDE = pr.read_bed(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed")
pvals = enricher.findEnriched(globallyDE, background=consensuses, minGenes=1, maxGenes=1e6)
pvals.to_csv(paths.outputDir + "rnaseq/Survival/pancancer_hallmarkGenes_enrichs.csv")
# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
res = pvals
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765) if p < 0.05 else (0.5,0.5,0.5) for p in res["P(Beta > 0)"].values]
sns.barplot(x=res.index, y=-np.log10(res["P(Beta > 0)"].values), palette=colors)
plt.xlabel("Cancer hallmark")
plt.xlim(plt.xlim()[0], plt.xlim()[1])
plt.xticks(rotation=90, fontsize=10)
plt.hlines(-np.log10(0.05), plt.xlim()[0], plt.xlim()[1], color="red", linestyle="dashed")
plt.ylabel("-log10(P-value)")
plt.gca().set_aspect(np.ptp(plt.xlim())/np.ptp(plt.ylim())*0.5)
plt.savefig(paths.outputDir + "rnaseq/Survival/pancancer_hallmarkGenes_enrichs.pdf",
            bbox_inches="tight")