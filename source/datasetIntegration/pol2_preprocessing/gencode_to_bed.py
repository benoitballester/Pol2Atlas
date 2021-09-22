# %%
import pandas as pd
import pyranges as pr
import numpy as np
# Extension of the genomic context file given (in both sides, bp)
genomicContextExtent = 1000

gencode = pr.read_gtf("/scratch/pdelangen/projet_these/data/annotation/gencode.v38.annotation.gtf")
gencode = gencode.as_df()
# %%
transcripts = gencode[gencode["Feature"] == "transcript"]
# %%
bedGenicwithExtent = transcripts[["Chromosome", "Start", "End", "gene_id", "Strand"]]
bedGenicwithExtent["Start"] = np.maximum(bedGenicwithExtent["Start"] - genomicContextExtent, 0)
bedGenicwithExtent["End"] = bedGenicwithExtent["End"] + genomicContextExtent
bedGenicwithExtent.sort_values(by=["Chromosome", "Start"], inplace=True)
bedGenicwithExtent.to_csv("/scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed",
                          sep="\t", header=None, index=None)
# %%
