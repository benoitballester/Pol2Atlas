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
from lib.pyGREATnewInput import pyGREAT as pyGREATglm
enricherglm = pyGREATglm("/scratch/pdelangen/projet_these/data_clean/GO_files/hsapiens.GO:BP.name.gmt",
                          geneFile=paths.gencode)
# %%

# %%
inclust = clusts == 4
queryClust = pr.PyRanges(consensuses[inclust])
# queryClust = pr.read_bed(paths.outputDir + "rnaseq/Survival2/globally_prognostic.bed")
pvals = enricherglm.findEnriched(queryClust, background=consensusesPr)
enricherglm.plotEnrichs(pvals)
# enricherglm.revigoTreemap(pvals, output="test.pdf")
# %%
class distPlotter:
    def __init__(self, gencode, query):
        gencodeChr = dict(tuple(gencode.groupby(0)))
        consensusesCpy = query.copy()
        consensusesCpy["Center"] = consensusesCpy["End"] * 0.5 + consensusesCpy["Start"] * 0.5
        allDists = []
        geneSizes = []
        for i, r in consensusesCpy.iterrows():
            try:
                chrom = gencodeChr[r["Chromosome"]][[1, 2, 4]].values
                diff = (chrom[:, [0,1]] - r["Center"])
                diff = np.where((chrom[:, 2] == "+")[:, None], diff, diff[:,::-1])
                allDists.append(diff)
            except KeyError:
                continue
        self.allDists = np.concatenate(allDists)

    def plot(self, xmin, xmax, tss_size=10000):
        positions = np.where(self.allDists[:, 0] <= 0, self.allDists[:, 0], 
                                                  self.allDists[:, 1] + tss_size)
        plt.hist(positions,np.linspace(xmin, xmax,200))
        return self

# %%
gencode = pd.read_csv("gencode_tx.bed", header=None, sep="\t", index_col=3)
dst = distPlotter(gencode, consensuses)
dst.plot(-1e5,1e5,10000)
# %%
test = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal/globally_DE.bed").as_df()
dst2 = distPlotter(gencode, test).plot(-1e5,1e5,10000)