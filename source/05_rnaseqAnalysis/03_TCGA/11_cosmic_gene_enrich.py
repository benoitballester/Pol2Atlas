# %%
import pandas as pd
import numpy as np
from settings import params, paths
# For COSMIC census data only
""" tab = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_all/cosmic_hallmarks/Cancer_Gene_Census_Hallmarks_Of_Cancer.tsv", 
                  sep="\t", encoding = "ISO-8859-1")

hallmarks = tab.groupby("HALLMARK")

with open(paths.tempDir + "hallmark_cancer.gmt", "w") as f:
    for g in hallmarks.groups:
        genes = tab.loc[hallmarks.groups[g], "GENE_NAME"].values
        genes = np.unique(genes)
        txt = "\t".join(genes)
        txt = g + "\t" + g + "\t" + txt + "\n"
        f.write(txt)

from lib.pyGREATglm import pyGREAT
enricher = pyGREAT(paths.tempDir + "hallmark_cancer.gmt",
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)

import pyranges as pr
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
globallyDE = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal/globally_DE.bed")
pvals = enricher.findEnriched(globallyDE, background=consensuses)

consensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
globallyDE = pr.read_bed(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed")
pvals = enricher.findEnriched(globallyDE, background=consensuses)
 """
# %%
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7013921/
tab = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_all/cosmic_hallmarks/hallmarks_chg.csv", 
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
