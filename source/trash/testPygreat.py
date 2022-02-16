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

consensuses = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
consensuses = consensuses[[0,1,2]]
consensuses["Name"] = np.arange(len(consensuses))
consensuses.columns = ["Chromosome", "Start", "End", "Name"]
consensusesPr = pr.PyRanges(consensuses)
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)

# %%
from lib.pyGREATglm import pyGREAT as pyGREATglm
enricherglm = pyGREATglm(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")
# %%
inclust = clusts==6
queryClust = pr.PyRanges(consensuses[inclust])
# queryClust = pr.read_bed(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed")
pvals = enricherglm.findEnriched(queryClust, consensusesPr)
qvals = pvals.copy()
qvals.loc[:] = fdrcorrection(qvals)[1]
qvals.sort_values()[qvals < 0.05]

# %%
