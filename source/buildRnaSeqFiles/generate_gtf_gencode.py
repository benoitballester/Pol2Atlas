# %%
import pyranges as pr
import pandas as pd
import numpy as np
import sys
sys.path.append("./")
from settings import params, paths


chrInfo = pd.read_csv(paths.genomeFile, sep="\t", index_col=0, header=None)

consensuses = pr.read_bed(paths.tempDir + "Pol2_500.bed")
consensuses = consensuses.as_df()
consensuses = consensuses.iloc[:, :3]
consensuses["gene_biotype"] = "protein_coding"
consensuses["Feature"] = "exon"
consensuses["Source"] = "Pol2"
endFix = []
for i in range(len(consensuses)):
    maxLen = chrInfo.loc[consensuses.iloc[i, 0]].values[0]
    endFix.append(min(consensuses.iloc[i, 2], maxLen))
consensuses["End"] = endFix
# Fix chromosome names
fixed = []
for c in consensuses["Chromosome"]:
    if c == "chrM":
        fixed.append("MT")
        continue
    splitted = c.split("_")
    if len(splitted) == 1:
        fixed.append(c[3:])
        continue
    else:
        fixed.append(splitted[1].replace("v", "."))
        continue
consensuses["Chromosome"] = fixed
plusStrand = consensuses.copy()
plusStrand["Strand"] = "+"
negStrand = consensuses.copy()
negStrand["Strand"] = "-"
consensuses = pd.concat([plusStrand, negStrand])
consensuses["gene_id"] = np.arange(len(consensuses))
consensuses["transcript_id"] = np.arange(len(consensuses))
consensuses["gene_name"] = np.arange(len(consensuses))

consensuses = pr.PyRanges(consensuses)
print(consensuses.to_gtf(paths.tempDir + "Pol2_500.gtf"))
# %%
