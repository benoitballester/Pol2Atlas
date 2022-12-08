# %%
import pandas
import os 
import sys
sys.path.append("./")
from settings import params, paths
import numpy as np
# Setup directories
countDir500 = paths.outputDir + "rnaseq/chagas_counts/500centroid"
try:
    os.mkdir(paths.outputDir + "rnaseq/chagas_counts")
    os.mkdir(countDir500)
except:
    pass
bamPath = "/shared/projects/pol2_chipseq/02.Alignment/"
folders = os.listdir(bamPath)


# %%
rule All:
    input:
        expand(countDir500+"/{fileIDs}.counts.summary", fileIDs=folders)

rule dlCount:
    output:
        countDir500+"/{fileIDs}.counts.summary", 
    threads:
        24
    resources:
        mem_mb=20000, disk_mb=20000
    shell:
        """
        mkdir {paths.tempDir}{wildcards.fileIDs}
        # Count reads in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2Hg19_500.saf \
           {countDir500}/{wildcards.fileIDs}.counts \
           {bamPath}{wildcards.fileIDs}/Aligned.sortedByCoord.out.bam\
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmp500/  
        
        # Delete temp file
        rm -rf {paths.tempDir}{wildcards.fileIDs}_tmp500
        """
