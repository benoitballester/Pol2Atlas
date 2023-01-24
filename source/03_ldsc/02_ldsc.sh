snakemake --snakefile source/ldsc/ldscPipeline.smk \
          --cores 54 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k --touch