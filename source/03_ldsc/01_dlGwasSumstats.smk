# %%
import pandas
import os
import urllib.request
import sys
sys.path.append("./")
from settings import params, paths

manifest = pandas.read_csv(f"{paths.ldscFilesPath}/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t")
lookupTable = pandas.read_csv(f"{paths.ldscFilesPath}/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t", header=None)[[0,1]]
availableFiles = set(lookupTable[0].values)
phenotypeList = []
downloads = []
nMiss = 0
for l in manifest.iterrows():
    row = l[1]
    phenoCode = row["phenotype"]
    if not type(phenoCode) == float and not phenoCode.endswith("raw"):
        if phenoCode in availableFiles:
            if row["ldsc_h2_significance"] in ["z7"]:
                try:
                    if not phenoCode in phenotypeList:
                        phenotypeList.append(phenoCode)
                        downloads.append(row["ldsc_sumstat_dropbox"])
                except KeyError:
                    nMiss += 1
                    continue

outputFiles = [f[:-4].split("/")[-1][:-1] for f in downloads]
dlEq = {}
for dl, out in zip(downloads, outputFiles):
    dlEq[out] = dl

try:
    os.mkdir(paths.ldscFilesPath + "ldsc_sumstats")
except:
    pass
# %%
rule all:
    input:
        expand(paths.ldscFilesPath + "ldsc_sumstats/{o}", o=outputFiles)


rule dlGWAS:
    threads:
        1
    output:
        paths.ldscFilesPath + "ldsc_sumstats/{outputFiles}"
    params:
        dlList = lambda wildcard: dlEq[wildcard[0]]
    shell:
       """
       wget {params.dlList} -O {paths.ldscFilesPath}ldsc_sumstats/{wildcards.outputFiles}; 
       """


