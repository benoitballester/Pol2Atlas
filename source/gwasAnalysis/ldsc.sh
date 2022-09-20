snakemake --snakefile source/gwasAnalysis/ldscPipeline.smk \
          --cores 32 \
          --rerun-incomplete \
          --use-singularity --singularity-args "-B /shared/projects/pol2_chipseq" \
          -k