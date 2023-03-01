snakemake --snakefile rerun_all/01_pol2_preprocessing.smk \
          --cores 54 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k

snakemake --snakefile rerun_all/02_datasetIntegration.smk \
          --cores 54 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k