snakemake --snakefile source/06_metacluster/02_ldscPipeline.smk \
          --cores 54 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k