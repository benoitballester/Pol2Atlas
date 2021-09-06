import pandas
import re
import urllib.request
from config.params import *
import os 


try:
    os.mkdir(parameters.tempDir + "liftedClusters/")
    os.mkdir(parameters.tempDir + "noLift/")
    os.mkdir(parameters.outputDir + "cluster_analysis/ldsc/")
except:
    pass


sumstatsFolder = parameters.dataPath + "GWAS/ldsc_sumstats/"
sumstatsFilesAll = os.listdir(sumstatsFolder)
sumstatsFiles = []
for i in sumstatsFilesAll:
    if i.endswith(".tsv.bgz"):
        sumstatsFiles.append(i)

tfFolder = parameters.outputDir + "cluster_analysis/clusterbed_matPOLR2A_Inter/"
tfClusts = os.listdir(tfFolder)
tfPaths = [tfFolder + f for f in tfClusts]

chroms = list(range(1,23))



rule all:
    input:
        expand(parameters.outputDir + "cluster_analysis/ldsc/{sumstatsFiles}.results", sumstatsFiles=sumstatsFiles)

annotFiles = [parameters.tempDir + f"ld.{c}.annot.gz" for c in chroms]

rule liftover:
    threads:
        1
    input:
        expand("{abc}", abc=tfPaths)
    output:
        parameters.tempDir + "liftedClusters/{tfClusts}"
    shell:
        """
            {parameters.dataPath}liftover/liftOver  \
            {parameters.outputDir}cluster_analysis/clusterbed_matPOLR2A_Inter/{wildcards.tfClusts} \
            {parameters.dataPath}liftover/hg38ToHg19.over.chain \
            {parameters.tempDir}liftedClusters/{wildcards.tfClusts} \
            {parameters.tempDir}noLift/{wildcards.tfClusts}
        """


rule makeAnnot:
    input:
        expand(parameters.tempDir + "liftedClusters/{tfClusts}", tfClusts=tfClusts)
    output:
        parameters.tempDir + "ld.22.annot.gz"
    shell:
        """
        python source/analysePol2Matrix/ldsc/makeCustomAnnot.py
        """

rule baselineLD:
    input:
        parameters.tempDir + "ld.22.annot.gz"
    threads:
        2
    singularity:
        parameters.dataPath + "singularity/s-ldsc/envSingularityLDSC.img"
    output:
        parameters.tempDir + "ld.{chroms}.l2.M_5_50"
    shell:
       """
       python source/analysePol2Matrix/ldsc/ldsc.py --l2 \
       --bfile {parameters.dataPath}ldsc/1000G_EUR_Phase3_plink/1000G.EUR.QC.{wildcards.chroms} \
       --ld-wind-cm 1 \
       --annot {parameters.tempDir}ld.{wildcards.chroms}.annot.gz  \
       --out {parameters.tempDir}ld.{wildcards.chroms}
       """

rule computeSLDSC:
    input:
        expand(parameters.tempDir + "ld.{chroms}.l2.M_5_50", chroms=chroms)
    threads:
        3
    singularity:
        parameters.dataPath + "singularity/s-ldsc/envSingularityLDSC.img"
    output:
        parameters.outputDir + "cluster_analysis/ldsc/{sumstatsFiles}.results"
    shell:
        """
        python source/analysePol2Matrix/ldsc/ldsc.py --h2 {sumstatsFolder}{wildcards.sumstatsFiles} \
        --ref-ld-chr {parameters.tempDir}ld. \
        --w-ld-chr {parameters.dataPath}ldsc/weights_hm3_no_hla/weights. \
        --annot {parameters.tempDir}ld. \
        --frqfile-chr {parameters.dataPath}ldsc/1000G_Phase3_frq/1000G.EUR.QC. \
        --out {parameters.outputDir}cluster_analysis/ldsc/{wildcards.sumstatsFiles} \
        --overlap-annot \
        --n-blocks 100000
        """