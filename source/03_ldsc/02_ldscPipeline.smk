import pandas
import re
import os
import urllib.request
import sys
sys.path.append("./")
from settings import params, paths

try:
    os.mkdir(paths.tempDir + "liftedClusters/")
    os.mkdir(paths.tempDir + "noLift/")
    os.mkdir(paths.outputDir + "/ldsc/")
except:
    pass


sumstatsFolder = paths.ldscFilesPath + "/"
sumstatsFilesAll = os.listdir(sumstatsFolder)
sumstatsFiles = []
for i in sumstatsFilesAll:
    if i.endswith(".tsv.bgz"):
        sumstatsFiles.append(i)

tfFolder = paths.outputDir + "clusters_bed/"
tfClusts = os.listdir(tfFolder)
tfPaths = [tfFolder + f for f in tfClusts]

chroms = list(range(1,23))


rule all:
    input:
        expand(paths.outputDir + "ldsc/{sumstatsFiles}.results", sumstatsFiles=sumstatsFiles)

annotFiles = [paths.tempDir + f"ld.{c}.annot.gz" for c in chroms]

rule liftover:
    threads:
        1
    input:
        expand("{abc}", abc=tfPaths)
    output:
        paths.tempDir + "liftedClusters/{tfClusts}"
    shell:
        """
            {paths.liftoverPath}liftOver  \
            {paths.outputDir}clusters_bed/{wildcards.tfClusts} \
            {paths.liftoverPath}hg38ToHg19.over.chain \
            {paths.tempDir}liftedClusters/{wildcards.tfClusts} \
            {paths.tempDir}noLift/{wildcards.tfClusts}
        """


rule makeAnnot:
    input:
        expand(paths.tempDir + "liftedClusters/{tfClusts}", tfClusts=tfClusts)
    output:
        paths.tempDir + "ld.22.annot.gz"
    shell:
        """
        python lib/ldsc/makeCustomAnnot.py
        """

rule baselineLD:
    input:
        paths.tempDir + "ld.22.annot.gz"
    threads:
        4
    singularity:
        paths.ldscSingularity
    output:
        paths.tempDir + "ld.{chroms}.l2.M_5_50"
    shell:
       """
       python lib/ldsc/ldsc.py --l2 \
       --bfile {paths.ldscFilesPath}/1000G_EUR_Phase3_plink/1000G.EUR.QC.{wildcards.chroms} \
       --ld-wind-cm 1 \
       --annot {paths.tempDir}ld.{wildcards.chroms}.annot.gz  \
       --out {paths.tempDir}ld.{wildcards.chroms}
       """

rule computeSLDSC:
    input:
        expand(paths.tempDir + "ld.{chroms}.l2.M_5_50", chroms=chroms)
    threads:
        6
    singularity:
        paths.ldscSingularity
    output:
        paths.outputDir + "ldsc/{sumstatsFiles}.results"
    shell:
        """
        python lib/ldsc/ldsc.py --h2 {sumstatsFolder}{wildcards.sumstatsFiles} \
        --ref-ld-chr {paths.tempDir}ld. \
        --w-ld-chr {paths.ldscFilesPath}/weights_hm3_no_hla/weights. \
        --annot {paths.tempDir}ld. \
        --frqfile-chr {paths.ldscFilesPath}/1000G_Phase3_frq/1000G.EUR.QC. \
        --out {paths.outputDir}ldsc/{wildcards.sumstatsFiles} \
        --overlap-annot \
        --n-blocks 100000
        """