# %%
import pandas
import sys
sys.path.append('lib')
from common import *
import os 
import numpy
from copy import deepcopy
from statsmodels.stats.multitest import fdrcorrection


lookupTable = pandas.read_csv(parameters.dataPath + "GWAS/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t", header=None)
lookupTable.index = lookupTable[0]
lookupTable = lookupTable.drop([0],1)
toReplace = pandas.isna(lookupTable[1])
lookupTable[pandas.isna(lookupTable[1])] = lookupTable[pandas.isna(lookupTable[1])].index[:,None]

resultsPath = parameters.outputDir + "cluster_analysis/ldsc/"
files = os.listdir(resultsPath)
results = []
for f in files:
    if f.endswith(".results"):
        results.append(resultsPath + f)
# %%
f = results[0]
traitName = lookupTable.loc[f.split("/")[-1].split(".")[0]].values[0]
if type(traitName) == np.ndarray:
    traitName = traitName[0]
tab = pandas.read_csv(f, sep="\t")
# enrichment
resultsEnrich = tab[["Enrichment"]]
resultsEnrich.index = [c.split("/")[-1].split(".")[0] for c in tab["Category"]]
resultsEnrich.columns = [traitName]
# pvals
resultsPval = tab[["Enrichment_p"]]
resultsPval.index = [c.split("/")[-1].split(".")[0] for c in tab["Category"]]
resultsPval.columns = [traitName]
# %%
for f in results[1:]:
    traitName = lookupTable.loc[f.split("/")[-1].split(".")[0]].values[0]
    if type(traitName) == np.ndarray:
        traitName = traitName[0]
    tab = pandas.read_csv(f, sep="\t")
    # enrichment
    gwasEnrich = tab[["Enrichment"]]
    gwasEnrich.index = [c.split("/")[-1].split(".")[0] for c in tab["Category"]]
    gwasEnrich.columns = [traitName]
    # pvals
    gwasPval = tab[["Enrichment_p"]]
    gwasPval.index = [c.split("/")[-1].split(".")[0] for c in tab["Category"]]
    gwasPval.columns = [traitName]
    resultsEnrich[traitName] = gwasEnrich
    resultsPval[traitName] = gwasPval
    
# %%
resultsQval = deepcopy(resultsPval)
# resultsQval.loc[:] = fdrcorrection(resultsQval.values.astype("float").flatten(), method="negcorr")[1].reshape(resultsQval.shape)
resultsQval.loc[:] = resultsQval*np.prod(resultsQval.shape)

# %%
resultsQval.to_csv(parameters.outputDir + "cluster_analysis/ldscResults/qvalues.tsv",sep="\t")
resultsEnrich.to_csv(parameters.outputDir + "cluster_analysis/ldscResults/enrich.tsv",sep="\t")
resultsPval.to_csv(parameters.outputDir + "cluster_analysis/ldscResults/pvalues.tsv",sep="\t")
# %%
