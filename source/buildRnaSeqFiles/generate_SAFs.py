# %%
import pandas as pd
import numpy as np
import numba
import sys
sys.path.append("./")
from settings import params, paths
import lib.utils.utils as utils
# %%
utils.runScript(paths.bedtoolsPath, 
               ["complement", "-i", paths.tempDir + "genicRegions_gc38.bed", "-g", paths.genomeFile], 
               f"{paths.tempDir}/intergenicRegions_gc38.bed")
# %%
# Read and convert bed to SAF
inputFile = paths.outputDir + "consensuses.bed"
filePrefix = "Pol2_Interg"

binSize = 10
flankingWidth = 0
normalizedWidth = 500

f = pd.read_csv(inputFile, sep="\t", header=None, usecols=[0,1,2,6,7])
f[3] = f.index
f = f[[3,0,1,2,6,7]]
df = f[[3,0,1,2]]
df.columns = ["GeneID", "Chr", "Start", "End"]
df["Strand"] = "."
df.to_csv(paths.tempDir + f"{filePrefix}.saf", sep="\t", index=False)

# %%
values = f.values
def createSubBinning(vals):
    centers = vals[:, 4] * 0.5 + vals[:, 5] * 0.5
    newStarts = (np.floor((centers-normalizedWidth)/binSize)*binSize).astype(int) - flankingWidth
    newEnds = (np.floor((centers+normalizedWidth)/binSize)*binSize).astype(int) + flankingWidth
    allBinsPerpeak = []
    for i in range(len(newStarts)):
        correspondingPeak = vals[i]
        if newStarts[i] >= 0:
            steps = np.arange(newStarts[i], newEnds[i]+binSize, binSize)
            allBins = np.zeros((len(steps) - 1, 5), dtype="object")
            allBins[:, 0] = [str(correspondingPeak[0]) + "_" + str(i) for i in range(len(steps)-1)] 
            allBins[:, 1] = [correspondingPeak[1]] * (len(steps) - 1)
            allBins[:, 2] = steps[:-1]
            allBins[:, 3] = np.roll(steps, -1)[:-1] - 1
            allBins[:, 4] = ["."]*(len(steps) - 1)
            allBinsPerpeak.append(allBins)
    return np.concatenate(allBinsPerpeak, axis=0)


# %%
# SAF file with 10bp bins at +-500bp Pol2 centroid
samplePoints = pd.DataFrame(createSubBinning(values))
samplePoints.columns = ["GeneID", "Chr", "Start", "End", "Strand"]
print(samplePoints)
samplePoints.to_csv(paths.tempDir + f"{filePrefix}_windowed.saf", sep="\t", index=False)
# %%
# Stranded
samplePoints["Strand"] = "-"
samplePoints.to_csv(paths.tempDir + f"{filePrefix}_windowed_minus.saf", sep="\t", index=False)
samplePoints["Strand"] = "+"
samplePoints.to_csv(paths.tempDir + f"{filePrefix}_windowed_plus.saf", sep="\t", index=False)
# %%
# SAF file at +-500bp Pol2 centroid
vals = values
centers = vals[:, 4] * 0.5 + vals[:, 5] * 0.5
newStarts = (np.floor((centers-normalizedWidth)/binSize)*binSize).astype(int) - flankingWidth
newEnds = (np.floor((centers+normalizedWidth)/binSize)*binSize).astype(int) + flankingWidth - 1
dfCopy = df
dfCopy["Start"] = newStarts
dfCopy["End"] = newEnds
dropped = (dfCopy["Start"] < 0)
utils.dump(dropped, paths.tempDir + "droppedInterg")
dfCopy = dfCopy.drop(dfCopy.loc[dropped].index)
dfCopy.to_csv(paths.tempDir + f"{filePrefix}_500.saf", sep="\t", index=False)
dfCopy[["Chr", "Start", "End"]].to_csv(paths.tempDir + f"{filePrefix}_500.bed", sep="\t", header=False, index=False)
# %%
# Shuffle +- 500bp Pol2 
shuffleCmd = f"""{paths.bedtoolsPath} shuffle -excl {paths.tempDir}/{filePrefix}_500.bed \
                                            -incl {paths.tempDir}/intergenicRegions_gc38.bed \
                                            -maxTries 10000 \
                                            -noOverlapping \
                                            -seed 42 \
                                            -chrom \
                                            -i {paths.tempDir}/{filePrefix}_500.bed \
                                            -g {paths.genomeFile} """
output = f"{paths.tempDir}/backgroundReg.bed"
splittedCmd = shuffleCmd.split()
utils.runScript(splittedCmd[0], 
               splittedCmd[1:], 
               output)                        
# %%
# Convert to SAF
f = pd.read_csv(output, sep="\t", header=None)
f.columns = ["Chr", "Start", "End"]
f.index.name = "GeneID"
f.to_csv(f"{paths.tempDir}/backgroundReg.saf", sep="\t")
# %%
