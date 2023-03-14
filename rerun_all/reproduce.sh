breachPath=/shared/projects/pol2_chipseq
snakemake --snakefile rerun_all/01_pol2_preprocessing.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k

snakemake --snakefile rerun_all/02_datasetIntegration.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k

snakemake --snakefile source/03_ldsc/01_dlGwasSumstats.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k

snakemake --snakefile source/03_ldsc/02_ldscPipeline.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k
# Download and count here
# sh source/05_rnaseqAnalysis/01_downloadCount/download-encode.sh
# sh source/05_rnaseqAnalysis/01_downloadCount/download-gtex.sh
# sh source/05_rnaseqAnalysis/01_downloadCount/download-tcga.sh
snakemake --snakefile rerun_all/05_02_GTEx-encode.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k

snakemake --snakefile rerun_all/05_03_TCGA.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B $breachPath" \
          -k