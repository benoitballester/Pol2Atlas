import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils


rule all:
    input:
        paths.outputDir + "descriptivePlots/distrib_overlap_per_peak.pdf"

rule integration:
    input:
        paths.outputDir + "filteredqval_dataset_interg.bed"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0201.txt"
    shell:
        "python source/02_datasetIntegration/01_integrativeAnalysis.py"

rule generateFiles:
    input:
        paths.tempDir + "end0201.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "Pol2_500.gtf"
    shell:
        """
        python source/04_buildRnaSeqFiles/01_generate_SAFs.py
        python source/04_buildRnaSeqFiles/02_generate_SAFs_hg19.py
        python source/04_buildRnaSeqFiles/03_generate_gtf.py
        """

rule intersectRef:
    input:
        paths.tempDir + "Pol2_500.gtf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "intersections_databases/rep_family.pdf"
    shell:
        "python source/02_datasetIntegration/02_intersect_table.py"

rule profiles:
    input:
        paths.outputDir + "intersections_databases/rep_family.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "/epigenetic/consRandom.pdf"
    shell:
        "python source/02_datasetIntegration/03_overlapChIPHistone.py"

rule roadmap:
    input:
        paths.outputDir + "/epigenetic/consRandom.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0204.txt"
    shell:
        "python source/02_datasetIntegration/04_overlapRoadmap.py"

rule additionalPlots:
    input:
        paths.tempDir + "end0204.txt"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "descriptivePlots/distrib_overlap_per_peak.pdf"
    shell:
        "python source/02_datasetIntegration/05_pol2_plots.py"