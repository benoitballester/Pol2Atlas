# %%
import pandas
import re
import urllib.request
from config.params import *
import os 

manifest = pandas.read_csv(parameters.dataPath + "tcga/BRCA_metadata/gdc_sample_sheet.2021-06-22.tsv", sep="\t")

countDir10 = parameters.outputDir + "counts/10bp"
countDir500 = parameters.outputDir + "counts/500centroid"
countDir = parameters.outputDir + "counts/polII"
try:
    os.mkdir(parameters.outputDir + "counts")
    os.mkdir(countDir10)
    os.mkdir(countDir500)
    os.mkdir(countDir)
except:
    pass

fileIDs = manifest["File ID"]
idNameMap = dict(zip(fileIDs, manifest["File Name"]))
tokenPath = parameters.dataPath + "tcga/token.txt"

rule All:
    input:
        expand(countDir10+"/{fileIDs}.txt.gz", fileIDs=fileIDs), expand(countDir500+"/{fileIDs}.txt.gz", fileIDs=fileIDs), expand(countDir+"/{fileIDs}.txt.gz", fileIDs=fileIDs)

rule dlCount:
    output:
        countDir10+"/{fileIDs}.txt.gz", countDir500+"/{fileIDs}.txt.gz", countDir+"/{fileIDs}.txt.gz"
    threads:
        1
    params:
        nameID = lambda wildcard : idNameMap[wildcard[0]]
    shell:
        """
        # Download bam file
        {parameters.dataPath}tcga/gdc-client download {wildcards.fileIDs} \
        -t {tokenPath} -d {parameters.dataPath}tcga/counts/temp/

        # Count reads in 10bp intervals
        {parameters.dataPath}tcga/featureCounts -T 1 -F SAF -O \
        --largestOverlap -a {parameters.tempDir}POLR2A_Inter_windowed.saf \
        -p -o {countDir10}/{wildcards.fileIDs}.counts \
        {parameters.dataPath}tcga/counts/temp/{wildcards.fileIDs}/{params.nameID}
        # Cut last column, gunzip and erase original count file
        cut -f 7 {countDir10}/{wildcards.fileIDs}.counts > {countDir10}/{wildcards.fileIDs}.txt
        gzip {countDir10}/{wildcards.fileIDs}.txt
        rm {countDir10}/{wildcards.fileIDs}.counts 
       
        # Count reads in +-500bp around centroid
        {parameters.dataPath}tcga/featureCounts -T 1 -F SAF -O \
        --largestOverlap -a {parameters.tempDir}POLR2A_Inter_500.saf \
        -p -o {countDir500}/{wildcards.fileIDs}.counts \
        {parameters.dataPath}tcga/counts/temp/{wildcards.fileIDs}/{params.nameID}
        # Cut last column, gunzip and erase original count file
        cut -f 7 {countDir500}/{wildcards.fileIDs}.counts > {countDir500}/{wildcards.fileIDs}.txt
        gzip {countDir500}/{wildcards.fileIDs}.txt
        rm {countDir500}/{wildcards.fileIDs}.counts 

        # Count read in consensus
        {parameters.dataPath}tcga/featureCounts -T 1 -F SAF -O \
        --largestOverlap -a {parameters.tempDir}POLR2A_Inter.saf \
        -p -o {countDir}/{wildcards.fileIDs}.counts \
        {parameters.dataPath}tcga/counts/temp/{wildcards.fileIDs}/{params.nameID}
        # Cut last column, gunzip and erase original count file
        cut -f 7 {countDir}/{wildcards.fileIDs}.counts > {countDir}/{wildcards.fileIDs}.txt
        gzip {countDir}/{wildcards.fileIDs}.txt
        rm {countDir}/{wildcards.fileIDs}.counts 

        # Delete bam file
        rm -rf {parameters.dataPath}tcga/counts/temp/{wildcards.fileIDs}
        """

