# %%
import pandas
import os 
import sys
sys.path.append("./")
from settings import params, paths
import numpy as np
# Setup directories
countDir10 = paths.outputDir + "rnaseq/encode_counts/10bp"
countDir500 = paths.outputDir + "rnaseq/encode_counts/500centroid"
countDir = paths.outputDir + "rnaseq/encode_counts/polII"
countDirBg = paths.outputDir + "rnaseq/encode_counts/BG"
countDirAll = paths.outputDir + "rnaseq/encode_counts/All_500"
try:
    os.mkdir(paths.outputDir + "rnaseq/encode_counts")
    os.mkdir(countDir10)
    os.mkdir(countDir500)
    os.mkdir(countDir)
    os.mkdir(countDirBg)
    os.mkdir(countDirAll)
except:
    pass

manifest = pandas.read_csv(paths.encodeMetadata, sep="\t")
manifest = manifest[(manifest["File assembly"] == "GRCh38") & (manifest["Output type"] == "alignments")]
# Order protocol priority
prio = ['Lab custom GRCh38', 'ENCODE3 GRCh38 V24', 'ENCODE4 v1.1.0 GRCh38 V29',
       'ENCODE4 v1.2.1 GRCh38 V29', 'ENCODE4 v1.2.3 GRCh38 V29']
# Experiments have been done on different processing pipelines, select the most recent one
experiments = manifest.groupby("Experiment accession").groups
kept = []
for i in experiments:
    prioExp = []
    for j in manifest.loc[experiments[i]]["File analysis title"]:
        prioExp.append(prio.index(j))
    prioExp = np.array(prioExp)
    highestPrio = np.max(prioExp)
    kept += list(np.array(experiments[i])[prioExp == highestPrio])
df = manifest.loc[kept]
fileIDs = df["File accession"]
idNameMap = dict(zip(fileIDs, df["File download URL"]))


rule All:
    input:
        expand(countDir10+"/{fileIDs}.counts.summary", fileIDs=fileIDs), 
        expand(countDir500+"/{fileIDs}.counts.summary", fileIDs=fileIDs), 
        expand(countDir+"/{fileIDs}.counts.summary", fileIDs=fileIDs),
        expand(countDirBg+"/{fileIDs}.counts.summary", fileIDs=fileIDs),
        expand(countDirAll+"/{fileIDs}.counts.summary", fileIDs=fileIDs)
rule dlCount:
    output:
        countDir10+"/{fileIDs}.counts.summary", 
        countDir500+"/{fileIDs}.counts.summary", 
        countDir+"/{fileIDs}.counts.summary",
        countDirBg+"/{fileIDs}.counts.summary",
        countDirAll+"/{fileIDs}.counts.summary",
    threads:
        2
    params:
        nameID = lambda wildcard : idNameMap[wildcard[0]]
    shell:
        """
        mkdir {paths.tempDir}{wildcards.fileIDs}
        # Download bam file
        wget -O {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
              --no-verbose \
              {params.nameID}  
        # Count reads in 10bp intervals in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_windowed.saf \
           {countDir10}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmpw/


       
        # Count reads in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_500.saf \
           {countDir500}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmp500/


        # Count reads in +-500bp around centroid (all Pol2)
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2.saf \
           {countDir}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmpall/

        # Count reads in consensus
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_all_500.saf \
           {countDirAll}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmpall500/


        # Count reads at random locations
        # Non-gencode v38 transcript and non-Pol2
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}backgroundReg.saf \
           {countDirBg}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{wildcards.fileIDs}.bam \
           {paths.featureCountPath} \
           {paths.tempDir}{wildcards.fileIDs}_tmpBG/
        
        

        # Delete bam file
        rm -rf {paths.tempDir}/{wildcards.fileIDs}
        """
