import pandas
import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils

tfFolder = paths.outputDir + "rnaseq/metacluster_10pct/markerCount/"
sumstatsFolder = paths.ldscFilesPath + "ldsc_sumstats/"
tempDir = paths.tempDir + "ldschalf/"
outputDir = paths.outputDir + "ldsc_meta/"

utils.createDir(outputDir)
utils.createDir(tempDir)
utils.createDir(tempDir + "liftedClusters/")
utils.createDir(tempDir + "noLift/")


sumstatsFilesAll = os.listdir(sumstatsFolder)
sumstatsFiles = []
for i in sumstatsFilesAll:
    if i.endswith(".tsv.bgz"):
        sumstatsFiles.append(i)


tfClusts = os.listdir(tfFolder)
tfClusts = [f for f in tfClusts if f.startswith("halfDatasets_")]
tfPaths = [tfFolder + f for f in tfClusts]

chroms = list(range(1,23))

rule all:
    input:
        expand(outputDir + "/{sumstatsFiles}.results", sumstatsFiles=sumstatsFiles)

# annotFiles = [tempDir + f"ld.{c}.annot.gz" for c in chroms]

rule liftover:
    threads:
        1
    input:
        expand("{abc}", abc=tfPaths)
    output:
        tempDir + "liftedClusters/{tfClusts}"
    shell:
        """
            {paths.liftoverPath}liftOver  \
            {tfFolder}{wildcards.tfClusts} \
            {paths.liftoverPath}hg38ToHg19.over.chain \
            {tempDir}liftedClusters/{wildcards.tfClusts} \
            {tempDir}noLift/{wildcards.tfClusts}
        """


rule makeAnnot:
    input:
        expand(tempDir + "liftedClusters/{tfClusts}", tfClusts=tfClusts)
    output:
        tempDir + "ld.22.annot.gz"
    shell:
        """
        python lib/ldsc/makeCustomAnnot.py {tempDir}"liftedClusters/" {tempDir}
        """

rule baselineLD:
    input:
        tempDir + "ld.22.annot.gz"
    threads:
        4
    singularity:
        paths.ldscSingularity
    output:
        tempDir + "ld.{chroms}.l2.M_5_50"
    shell:
       """
       python lib/ldsc/ldsc.py --l2 \
       --bfile {paths.ldscFilesPath}/1000G_EUR_Phase3_plink/1000G.EUR.QC.{wildcards.chroms} \
       --ld-wind-cm 1 \
       --annot {tempDir}ld.{wildcards.chroms}.annot.gz  \
       --out {tempDir}ld.{wildcards.chroms}
       """

rule computeSLDSC:
    input:
        expand(tempDir + "ld.{chroms}.l2.M_5_50", chroms=chroms)
    threads:
        6
    singularity:
        paths.ldscSingularity
    output:
        outputDir + "{sumstatsFiles}.results"
    shell:
        """
        python lib/ldsc/ldsc.py --h2 {sumstatsFolder}{wildcards.sumstatsFiles} \
        --ref-ld-chr {tempDir}ld. \
        --w-ld-chr {paths.ldscFilesPath}/weights_hm3_no_hla/weights. \
        --annot {tempDir}ld. \
        --frqfile-chr {paths.ldscFilesPath}/1000G_Phase3_frq/1000G.EUR.QC. \
        --out {outputDir}{wildcards.sumstatsFiles} \
        --overlap-annot \
        --n-blocks 1000000000
        """