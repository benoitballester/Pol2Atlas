import pandas
import re
import urllib.request
from settings import params, paths

manifest = pandas.read_csv(parameters.dataPath + "GWAS/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t")
lookupTable = pandas.read_csv(parameters.dataPath + "GWAS/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t", header=None)[[0,1]]
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

rule all:
    input:
        expand(parameters.dataPath + "GWAS/ldsc_sumstats/{o}", o=outputFiles)


rule dlGWAS:
    threads:
        1
    output:
        parameters.dataPath + "GWAS/ldsc_sumstats/{outputFiles}"
    params:
        dlList = lambda wildcard: dlEq[wildcard[0]]
    shell:
       """
       wget {params.dlList} -O {parameters.dataPath}GWAS/ldsc_sumstats/{wildcards.outputFiles}; 
       """


