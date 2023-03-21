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
pvals = enricher.findEnriched(globallyDE, background=consensuses)
enricher.plotEnrichs(pvals)
pvals.to_csv(paths.outputDir + "rnaseq/TumorVsNormal/pancancer_hallmarkGenes_enrichs.csv")
# %%
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
globallyDE = pr.read_bed(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed")
pvals = enricher.findEnriched(globallyDE, background=consensuses, minGenes=1, maxGenes=1e6)
pvals.to_csv(paths.outputDir + "rnaseq/Survival/pancancer_hallmarkGenes_enrichs.csv")

# %%
