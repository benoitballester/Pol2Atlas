# %%
import pandas
import os 
import sys
sys.path.append("./")
from settings import params, paths

# Setup count directories
countDir10 = paths.outputDir + "rnaseq/gtex_counts/10bp"
countDir500 = paths.outputDir + "rnaseq/gtex_counts/500centroid"
countDir = paths.outputDir + "rnaseq/gtex_counts/polII"
countDirBg = paths.outputDir + "rnaseq/gtex_counts/BG"
countDirAll = paths.outputDir + "rnaseq/gtex_counts/All_500"
try:
    os.mkdir(paths.outputDir + "rnaseq/gtex_counts")
    os.mkdir(countDir10)
    os.mkdir(countDir500)
    os.mkdir(countDir)
    os.mkdir(countDirBg)
    os.mkdir(countDirAll)
except:
    pass

manifest = pandas.read_json(paths.gtexData + "file-manifest.json")
manifest["object_id"] = [id.split("/")[1] for id in manifest["object_id"]]
fileIDs = manifest["object_id"]
fileNames = manifest["file_name"]
idNameMap = dict(zip(fileNames, fileIDs))

rule All:
    input:
        expand(countDir10+"/{fileIDs}.counts.summary", fileIDs=fileNames), 
        expand(countDir500+"/{fileIDs}.counts.summary", fileIDs=fileNames), 
        expand(countDir+"/{fileIDs}.counts.summary", fileIDs=fileNames),
        expand(countDirBg+"/{fileIDs}.counts.summary", fileIDs=fileNames),
        expand(countDirAll+"/{fileIDs}.counts.summary", fileIDs=fileNames)

rule createProfile:
    output:
        paths.gtexData + "profCheck.txt"
    shell:
        '''
        {paths.gtexData}gen3-client configure --profile=AnVIL --cred={paths.gtexData}credentials.json --apiendpoint=https://gen3.theanvil.io
        touch {paths.gtexData}profCheck.txt
        '''
        
rule dlCount:
    input:
        paths.gtexData + "profCheck.txt"
    output:
        countDir10+"/{fileNames}.counts.summary", 
        countDir500+"/{fileNames}.counts.summary", 
        countDir+"/{fileNames}.counts.summary",
        countDirBg+"/{fileNames}.counts.summary",
        countDirAll+"/{fileNames}.counts.summary",
    threads:
        4
    params:
        nameID = lambda wildcard : idNameMap[wildcard[0]]
    shell:
        """
        # Download bam file
        # Redirect to dev/null because it would write terabytes of loading bars
        echo y | {paths.gtexData}/gen3-client download-single \
            --guid={params.nameID} \
            --profile=AnVIL \
            --download-path={paths.tempDir}{params.nameID}/ \
            --protocol=s3 

        # Count reads in 10bp intervals in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_windowed.saf \
           {countDir10}/{wildcards.fileNames}.counts \
           {paths.tempDir}{params.nameID}/{wildcards.fileNames} \
           {paths.featureCountPath} \
           {paths.tempDir}{params.nameID}_tmpw/ \
           
       
        # Count reads in +-500bp around centroid
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_500.saf \
           {countDir500}/{wildcards.fileNames}.counts \
           {paths.tempDir}{params.nameID}/{wildcards.fileNames} \
           {paths.featureCountPath} \
           {paths.tempDir}{params.nameID}_tmp500/

        # Count reads in +-500bp around centroid (all Pol2)
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2.saf \
           {countDir}/{wildcards.fileNames}.counts \
           {paths.tempDir}{params.nameID}/{wildcards.fileNames} \
           {paths.featureCountPath}\
           {paths.tempDir}{params.nameID}_tmpall/

        # Count reads in consensus
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}Pol2_all_500.saf \
           {countDirAll}/{wildcards.fileNames}.counts \
           {paths.tempDir}{params.nameID}/{wildcards.fileNames} \
           {paths.featureCountPath}\
           {paths.tempDir}{params.nameID}_tmpall500/


        # Count reads at random locations
        # Non-gencode v38 transcript and non-Pol2
        sh source/rnaseqAnalysis/downloadCount/readCountsAtlas.sh \
           {paths.tempDir}backgroundReg.saf \
           {countDirBg}/{wildcards.fileNames}.counts \
           {paths.tempDir}{params.nameID}/{wildcards.fileNames} \
           {paths.featureCountPath}\
           {paths.tempDir}{params.nameID}_tmpBG/
        
        # Delete bam file
        rm -rf {paths.tempDir}/{params.nameID}
        """
