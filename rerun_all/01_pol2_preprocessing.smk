import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils

utils.createDir(paths.outputDir)
utils.createDir(paths.tempDir)
utils.createDir(paths.outputDir + "peaksPerDataset/")


rule all:
    input:
        paths.outputDir + "filteredqval_dataset_all.bed"

rule gencode_to_bed:
    input:
        paths.gencode
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "intragenicRegions_gc38.bed"
    shell:
        "python source/01_pol2_preprocessing/01_gencode_to_bed.py"

rule removeBlacklisted:
    input:
        paths.gencode
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "filtered.bed"
    shell:
        f"sh source/01_pol2_preprocessing/02_removeBlacklisted.sh {paths.bedtoolsPath} \
                                                                   {paths.allPol2File} \
                                                                   {paths.encodeBlacklist}\
                                                                   {paths.tempDir}filtered.bed"

rule removeGenic:
    input:
        [paths.tempDir + "intragenicRegions_gc38.bed", paths.tempDir + "filtered.bed"]
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "filtered_interg.bed"
    shell:
        f"sh source/01_pol2_preprocessing/03_remove_transcripts.sh {paths.bedtoolsPath} \
                                                                   {paths.tempDir}filtered.bed \
                                                                   {paths.tempDir}genicRegions_gc38.bed\
                                                                   {paths.tempDir}filtered_interg.bed"

rule splitFilter:
    input:
        paths.tempDir + "filtered_interg.bed"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "filteredqval_dataset_all.bed"
    shell:
        f"python source/01_pol2_preprocessing/05_splitByExperiment_Interg.py {paths.tempDir}filtered_interg.bed \
                                                                   {paths.outputDir}peaksPerDataset/ \
                                                                   {paths.tempDir}filtered.bed\
                                                                   {paths.outputDir}filteredqval_dataset_all.bed \
                                                                   {paths.outputDir}filteredqval_dataset_interg.bed"
