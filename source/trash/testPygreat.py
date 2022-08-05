# %%
from lib import pyGREATglm as pyGREAT
import pandas as pd
import numpy as np
import scipy.stats as ss
import pyranges as pr
import sys
from statsmodels.stats.multitest import fdrcorrection
sys.path.append("./")
from settings import params, paths
from lib.utils import matrix_utils
import matplotlib.pyplot as plt

consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[[0,1,2]]
consensuses["Name"] = np.arange(len(consensuses))
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensuses)
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)
# %%
from lib.pyGREAT_topGenes import pyGREAT as pyGREATglm
enricherglm = pyGREATglm(paths.GOfile,
                          geneFile=paths.gencode,
                          chrFile=paths.genomeFile)
# %%
'''
class regLogicGREAT:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF, chrInfo):
        # Infered regulatory domain logic
        
        copyTx = txDF.copy()
        copyTx["Start"] = (txDF["Start"] - self.upstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] - self.downstream)
        copyTx["End"] = (txDF["Start"] + self.downstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] + self.upstream)
        copyTx.sort_values(["Chromosome", "Start"], inplace=True)
        gb = copyTx.groupby("Chromosome")
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = copyTx["Chromosome"] == c
            nextReg = np.roll(copyTx["Start"][inIdx], -1)
            nextReg[-1] = chrInfo.loc[c].values[0]
            previousReg = np.roll(copyTx["End"][inIdx], 1)
            previousReg[0] = 0
            nextReg[-1] = chrInfo.loc[c].values[0]
            extMin = np.maximum(copyTx["Start"][inIdx] - self.distal, previousReg)
            extMax = np.minimum(copyTx["End"][inIdx] + self.distal, nextReg)
            extMin = np.minimum(copyTx["Start"][inIdx], extMin)
            extMax = np.maximum(copyTx["End"][inIdx], extMax)
            copyTx.loc[copyTx["Chromosome"] == c, "Start"] = np.clip(extMin, 0, chrInfo.loc[c].values[0])
            copyTx.loc[copyTx["Chromosome"] == c, "End"] = np.clip(extMax, 0, chrInfo.loc[c].values[0])
        return copyTx


reg = regLogicGREAT(5000,1000,1000000)(enricherglm.txList, 
                                    pd.read_csv(paths.genomeFile, sep="\t", index_col=0, header=None))
        
reg.to_csv(paths.tempDir + "reggreat.bed", header=None, index=None, sep="\t")
'''
# %%
inclust = clusts == 6
queryClust = pr.PyRanges(consensuses[inclust])
# queryClust = pr.read_bed('/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/Survival2/globally_prognostic.bed')
pvals = enricherglm.findEnriched(queryClust, background=consensusesPr)
enricherglm.plotEnrichs(pvals)
enricherglm.clusterTreemap(pvals, score="-log10(pval)", metric="yule")
# %%
from scipy.stats import hypergeom
detected = np.intersect1d(list(pvals.index), enricherglm.mat.columns)
detectedMat = enricherglm.mat[detected]
N = detectedMat.shape[1]
sig = pvals.index[pvals["BH corrected p-value"] < 0.25]
sig = np.intersect1d(list(sig), detectedMat.columns)
n = len(sig)
stats = dict()
K = detectedMat.sum(axis=1)
k = detectedMat[sig].sum(axis=1)
p = hypergeom(N, K, n).sf(k-1)
stats = pd.DataFrame(p, index=detectedMat.index, columns=["Pvalue"])[(k > 0) & (K >= 3)]
# %%
stats["BH corrected p-value"] = 1.0
stats["BH corrected p-value"] = fdrcorrection(stats["Pvalue"])[1]
stats["-log10(pval)"] = -np.log10(stats["Pvalue"])
print(stats.sort_values("Pvalue")[:50])
enricherglm.clusterTreemap(stats, alpha=0.25, score="-log10(pval)", metric="yule")
# %%
class distPlotter:
    def __init__(self, gencode, query):
        gencodeChr = dict(tuple(gencode.groupby("Chromosome")))
        consensusesCpy = query.copy()
        consensusesCpy["Center"] = consensusesCpy["End"] * 0.5 + consensusesCpy["Start"] * 0.5
        allDists = []
        geneSizes = []
        for i, r in consensusesCpy.iterrows():
            try:
                chrom = gencodeChr[r["Chromosome"]][["Start", "End", "Strand"]].values
                diff = (chrom[:, [0,1]] - r["Center"])
                diff = np.where((chrom[:, 2] == "+")[:, None], diff, diff[:,::-1])
                allDists.append(diff)
            except:
                continue
        self.allDists = np.concatenate(allDists)

    def plot(self, xmin, xmax, tss_size=10000):
        positions = np.where(self.allDists[:, 0] <= 0, self.allDists[:, 0], 
                                                  self.allDists[:, 1] + tss_size)
        plt.hist(positions,np.linspace(xmin, xmax,200))
        return self

# %%
gencode = enricherglm.txList
dst = distPlotter(gencode, consensuses)
dst.plot(-1e5,1e5,10000)
# %%
test = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal2/globally_DE.bed").as_df()
dst2 = distPlotter(gencode, test).plot(-1e5,1e5,10000)
# %%
