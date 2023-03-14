import os
import sys
sys.path.append("./")
from settings import params, paths
from lib.utils import utils
utils.createDir(paths.outputDir + "rnaseq/")

rule all:
    input:
        paths.outputDir + "dist_to_genes/gtex_tail/balancedAccuraciesbarplot.pdf"

rule readDistrib:
    input:
        paths.tempDir + "end0204.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end050201.txt"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/01_plotReadDistribs.py"

rule dist_to_genes:
    input:
        paths.tempDir + "end050201.txt"
    singularity:
        paths.singularityImg
    output:
        paths.tempDir + "end050202.txt"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/02_dist_to_genes.py"

rule encodeRnaseq:
    input:
        paths.tempDir + "end050202.txt"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/encode_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/03_encode_rnaseq.py"

rule GTEx_main:
    input:
        paths.outputDir + "rnaseq/encode_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/gtex_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/04_GTex.py"

rule GTEx_genic:
    input:
        paths.outputDir + "rnaseq/gtex_rnaseq/HM_50Pol_encode_order_pol2_signal.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/05_GTEx_genes.py"

rule GTEx_eqtl:
    input:
        paths.outputDir + "rnaseq/gtex_rnaseq/umap_samples_gene_centric.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "eqtlAnalysis/markerCount.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/06_gtex_eqtl.py"

rule smallSampleSize:
    input:
        paths.outputDir + "eqtlAnalysis/markerCount.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/precisionRecall_t.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/07_heart_gtex_stat_power.py"

rule gtexNoTail:
    input:
        paths.outputDir + "rnaseq/gtex_rnaseq_heart_DE/precisionRecall_t.pdf"
    singularity:
        paths.singularityImg
    output:
        paths.outputDir + "dist_to_genes/gtex_tail/balancedAccuraciesbarplot.pdf"
    shell:
        "python source/05_rnaseqAnalysis/02_GTEx_encode/08_GTex_trimmed_tail.py"