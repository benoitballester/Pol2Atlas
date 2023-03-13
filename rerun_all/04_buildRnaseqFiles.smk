import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils


rule all:
    input:
        paths.tempDir + "Pol2_500.gtf"

rule generateFiles:
    input:
        paths.tempDir + "end0303.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "Pol2_500.gtf"
    shell:
        """
        python source/04_buildRnaSeqFiles/01_generateSAFs.py
        python source/04_buildRnaSeqFiles/02_generateSAFs_hg19.py
        python source/04_buildRnaSeqFiles/03_generate_gtf.py
        """