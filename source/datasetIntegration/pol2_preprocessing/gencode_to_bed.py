# %%
import pandas as pd
import pyranges as pr
import numpy as np
sys.path.append("./")
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
