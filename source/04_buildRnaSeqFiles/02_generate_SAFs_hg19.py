# %%
import pandas as pd
import numpy as np
import numba
import sys
sys.path.append("./")
from settings import params, paths
import lib.utils.utils as utils
import pyranges as pr

# %%
# Generate intergenic regions
utils.runScript(f"{paths.liftoverPath}liftOver", 
                [f"{paths.outputDir}consensuses.bed",
                f"{paths.liftoverPath}hg38ToHg19.over.chain",
                f"{paths.outputDir}consensusesHg19.bed",
                f"{paths.tempDir}noLift/noLiftInterg.bed"])
# %%   
# Read and convert bed to SAF
inputFile = paths.outputDir + "consensusesHg19.bed"
filePrefix = "Pol2Hg19"

binSize = 10
flankingWidth = 0
normalizedWidth = 500

f = pd.read_csv(inputFile, sep="\t", header=None, usecols=[0,1,2,3,6,7])
f = f[[3,0,1,2,6,7]]
df = f[[3,0,1,2]]
df.columns = ["GeneID", "Chr", "Start", "End"]
df["Strand"] = "."
df.to_csv(paths.tempDir + f"{filePrefix}.saf", sep="\t", index=False)

# %%
values = f.values
# %%
# SAF file at +-500bp Pol2 centroid

vals = values
centers = vals[:, 4] * 0.5 + vals[:, 5] * 0.5
newStarts = (np.floor((centers-normalizedWidth)/binSize)*binSize).astype(int) - flankingWidth
newEnds = (np.floor((centers+normalizedWidth)/binSize)*binSize).astype(int) + flankingWidth - 1
dfCopy = df
dfCopy["Start"] = np.maximum(newStarts, 0)
dfCopy["End"] = newEnds
# dropped = (dfCopy["Start"] < 0)
# utils.dump(dropped, paths.tempDir + "droppedInterg")
#dfCopy = dfCopy.drop(dfCopy.loc[dropped].index)
dfCopy.to_csv(paths.tempDir + f"{filePrefix}_500.saf", sep="\t", index=False)
dfCopy[["Chr", "Start", "End"]].to_csv(paths.tempDir + f"{filePrefix}_500.bed", sep="\t", header=False, index=False)

# %%
