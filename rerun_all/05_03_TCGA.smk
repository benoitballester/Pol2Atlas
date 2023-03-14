import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils
utils.createDir(paths.outputDir + "rnaseq/")

rule all:
    input:
        paths.tempDir + "end0509.txt"

rule allTCGA:
    input:
        paths.outputDir + "dist_to_genes/gtex_tail/balancedAccuraciesbarplot.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/TCGA2/HM_hv.pdf"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/01_All_tcga.py"

rule DE_analysis:
    input:
        paths.outputDir + "rnaseq/TCGA2/HM_hv.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0503.txt"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/02_NormalVsTumor.py"

rule survival:
    input:
        paths.tempDir + "end0503.txt"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/Survival/globally_prognostic_heatmap_positions.pdf"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/03_Survival.py"

rule genomicIntersect:
    input:
        paths.outputDir + "rnaseq/Survival/globally_prognostic_heatmap_positions.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "intersections_databases/gtex_heart_rep_family.pdf"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/04_intersect_pancancer.py"

rule drawPanCancerKaplan:
    input:
        paths.outputDir + "intersections_databases/gtex_heart_rep_family.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0505.txt"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/05_kaplanmeierplots.py"

rule subtypesBRCA:
    input:
        paths.tempDir + "end0505.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0506.txt"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/06_BRCA_subtypes.py"

rule subtypesTHCA:
    input:
        paths.tempDir + "end0506.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0507.txt"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/07_THCA_subtypes.py"

rule cosmicGSEA:
    input:
        paths.tempDir + "end0507.txt"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/Survival/pancancer_hallmarkGenes_enrichs.csv"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/08_cosmic_gene_enrich.py"

rule geneCorrelation:
    input:
        paths.outputDir + "rnaseq/Survival/pancancer_hallmarkGenes_enrichs.csv"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end0509.txt"
    shell:
        "python source/05_rnaseqAnalysis/03_TCGA/09_corrGene.py"