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
        paths.tempDir + "end0201.txt"

rule integration:
    input:
        paths.outputDir + "filteredqval_dataset_interg.bed"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0201.txt"
    shell:
        "python source/02_datasetIntegration/01_integrativeAnalysis.py"

