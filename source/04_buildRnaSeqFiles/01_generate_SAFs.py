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
utils.runScript(paths.bedtoolsPath, 
               ["complement", "-i", paths.tempDir + "genicRegions_gc38.bed", "-g", paths.genomeFile], 
               f"{paths.tempDir}/intergenicRegions_gc38.bed")
# %%
# Read and convert bed to SAF
inputFile = paths.outputDir + "consensuses.bed"
filePrefix = "Pol2"

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
dfCopy["Start"] = np.maximum(newStarts, 0)
dfCopy["End"] = newEnds
# dropped = (dfCopy["Start"] < 0)
# utils.dump(dropped, paths.tempDir + "droppedInterg")
#dfCopy = dfCopy.drop(dfCopy.loc[dropped].index)
dfCopy.to_csv(paths.tempDir + f"{filePrefix}_500.saf", sep="\t", index=False)
dfCopy[["Chr", "Start", "End"]].to_csv(paths.tempDir + f"{filePrefix}_500.bed", sep="\t", header=False, index=False)
# %%
# Cover the genome with 1kbp bins to estimate background noise
with open(paths.genomeFile, 'r') as f:
    chrLen = dict()
    genomeSize = 0
    for l in f:
        l = l.rstrip("\n").split("\t")
        chrLen[l[0]] = int(l[1])
        genomeSize += int(l[1])

randomChroms = []
for c in np.unique(dfCopy["Chr"]):
    binStart = np.arange(0, chrLen[c], 1000)
    binEnd = np.roll(binStart,-1)[:-1] - 1
    binStart = binStart[:-1]
    df = pd.DataFrame()
    df["Start"] = binStart
    df["End"] = binEnd
    df["Chromosome"] = c
    df = df[["Chromosome", "Start", "End"]]
    randomChroms.append(df)
randomBins = pr.PyRanges(pd.concat(randomChroms))

# %%
# Exclude Pol2, encode blacklist, genic regions from background
pol2 = dfCopy[["Chr", "Start", "End"]]
pol2.columns = ["Chromosome", "Start", "End"]
blackList = pd.read_csv(paths.encodeBlacklist, sep="\t", header=None)[[0,1,2]]
blackList.columns = ["Chromosome", "Start", "End"]
genic = pd.read_csv(paths.tempDir + "genicRegions_gc38.bed", sep="\t", header=None)[[0,1,2]]
genic.columns = ["Chromosome", "Start", "End"]
excluded = pr.PyRanges(pd.concat([pol2, blackList, genic]))

keptReg = randomBins.overlap(excluded, invert=True)
keptReg = keptReg.as_df()
# %%
# Convert to SAF
keptReg.columns = ["Chr", "Start", "End"]
keptReg["Strand"] = "."
keptReg.index.name = "GeneID"
keptReg.to_csv(f"{paths.tempDir}/backgroundReg.saf", sep="\t")
keptReg.to_csv(f"{paths.tempDir}/background.bed", sep="\t", index=None, header=None)
# %%
# All Pol II consensuses
allPol2ConsensusesDir = paths.allPol2Consensuses
f = pd.read_csv(allPol2ConsensusesDir, sep="\t", header=None, usecols=[0,1,2,6,7])
f[3] = f.index
f = f[[3,0,1,2,6,7]]
df = f[[3,0,1,2]]
df.columns = ["GeneID", "Chr", "Start", "End"]
df["Strand"] = "."
df.to_csv(paths.tempDir + f"Pol2_all.saf", sep="\t", index=False)
values = f.values
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
dfCopy.to_csv(paths.tempDir + f"Pol2_all_500.saf", sep="\t", index=False)
# %%
