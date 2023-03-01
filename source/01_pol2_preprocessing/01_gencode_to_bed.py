# %%
import sys

import numpy as np
import pandas as pd
import pyranges as pr

sys.path.append("./")
from lib.utils import utils
from settings import params, paths


# Extension of the genomic context file given (in both sides, bp)
genomicContextExtent = 1000
gencode = pr.read_gtf(paths.gencode)
gencode = gencode.as_df()
# %%
transcripts = gencode[gencode["Feature"] == "transcript"]
# %%
bedGenicwithExtent = transcripts[["Chromosome", "Start", "End", "gene_id", "Strand"]]
bedGenicwithExtent["Start"] = np.maximum(bedGenicwithExtent["Start"] - genomicContextExtent, 0)
bedGenicwithExtent["End"] = bedGenicwithExtent["End"] + genomicContextExtent
bedGenicwithExtent.sort_values(by=["Chromosome", "Start"], inplace=True)
bedGenicwithExtent.to_csv(paths.tempDir + "/genicRegions_gc38.bed",
                          sep="\t", header=None, index=None)
# %%
# Intragenic only
bedIntra = transcripts.copy()
bedIntra = bedIntra[["Chromosome", "Start", "End", "gene_id", "Strand"]]
bedIntra["Start"] += np.where(bedIntra["Strand"].values == "+", genomicContextExtent, 0)
bedIntra["End"] += np.where(bedIntra["Strand"].values == "-", -genomicContextExtent, 0)
bedIntra["Start"] = np.maximum(bedIntra["Start"], 0)
bedIntra.sort_values(by=["Chromosome", "Start"], inplace=True)
bedIntra = bedIntra[bedIntra["Start"] < bedIntra["End"]]
bedIntra.to_csv(paths.tempDir + "/intragenicRegions_gc38.bed",
                          sep="\t", header=None, index=None)
# %%
