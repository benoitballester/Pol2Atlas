import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils
utils.createDir(paths.outputDir + "rnaseq/")

rule all:
    input:
        paths.outputDir + "allMarkers_allIntersects.csv"

rule metacluster:
    input:
        paths.tempDir + "end0509.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0601.txt"
    shell:
        "python source/06_metacluster/01_metacluster_filter.py"

rule genMarkerTables:
    input:
        paths.tempDir + "end0601.txt"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "allMarkers_allIntersects.csv"
    shell:
        "python source/06_metacluster/02_marker_tables.py"
