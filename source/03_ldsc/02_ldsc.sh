snakemake --snakefile source/03_ldsc/02_ldscPipeline.smk \
          --cores 54 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k