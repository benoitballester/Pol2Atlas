snakemake --snakefile rerun_all/01_pol2_preprocessing.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k

snakemake --snakefile rerun_all/02_datasetIntegration.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k

snakemake --snakefile source/03_ldsc/01_dlGwasSumstats.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k \

snakemake --snakefile source/03_ldsc/02_ldscPipeline.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k

snakemake --snakefile rerun_all/04_buildRnaseqFiles.smk \
          --cores $1 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k
# Download and count should be here