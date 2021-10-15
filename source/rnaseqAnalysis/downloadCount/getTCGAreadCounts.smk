# %%
import pandas
import os 
import sys
sys.path.append("./")
from settings import params, paths


manifest = pandas.read_csv(paths.manifest, sep="\t")

countDir10 = paths.outputDir + "rnaseq/counts/10bp"
countDir500 = paths.outputDir + "rnaseq/counts/500centroid"
countDir = paths.outputDir + "rnaseq/counts/polII"
countDirBg = paths.outputDir + "rnaseq/counts/BG"
countDirAll = paths.outputDir + "rnaseq/counts/All_500"
try:
    os.mkdir(paths.outputDir + "rnaseq/counts")
    os.mkdir(countDir10)
    os.mkdir(countDir500)
    os.mkdir(countDir)
    os.mkdir(countDirBg)
    os.mkdir(countDirAll)
except:
    pass

fileIDs = manifest["id"]
idNameMap = dict(zip(fileIDs, manifest["filename"]))
# fileIDs = "d9c22b90-0792-4d33-a55a-64366814db98"
# idNameMap = dict(zip(["d9c22b90-0792-4d33-a55a-64366814db98"], ["b6ae6275-da9e-426c-891d-0d6a7b715d31_gdc_realn_rehead.bam"]))


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
        1
    params:
        nameID = lambda wildcard : idNameMap[wildcard[0]]
    shell:
        """
        # Download bam file
        # Redirect to dev/null because it would write terabytes of loading bars
        {paths.gdcClientPath} download {wildcards.fileIDs} \
                                 -t {paths.tokenPath} \
                                 -d {paths.tempDir} &> /dev/null

        # Count reads in 10bp intervals in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_windowed.saf \
           {countDir10}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{params.nameID} \
           {paths.featureCountPath}\
           {paths.tempDir}{wildcards.fileIDs}_tmpw/


       
        # Count reads in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_500.saf \
           {countDir500}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{params.nameID} \
           {paths.featureCountPath}\
           {paths.tempDir}{wildcards.fileIDs}_tmp500/


        # Count reads in +-500bp around centroid (all Pol2)
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2.saf \
           {countDir}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{params.nameID} \
           {paths.featureCountPath}\
           {paths.tempDir}{wildcards.fileIDs}_tmpall/

        # Count reads in consensus
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_all_500.saf \
           {countDirAll}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{params.nameID} \
           {paths.featureCountPath}\
           {paths.tempDir}{wildcards.fileIDs}_tmpall500/


        # Count reads at random locations
        # Non-gencode v38 transcript and non-Pol2
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}backgroundReg.saf \
           {countDirBg}/{wildcards.fileIDs}.counts \
           {paths.tempDir}{wildcards.fileIDs}/{params.nameID} \
           {paths.featureCountPath}\
           {paths.tempDir}{wildcards.fileIDs}_tmpBG/
        
        

        # Delete bam file
        rm -rf {paths.tempDir}/{wildcards.fileIDs}
        """
