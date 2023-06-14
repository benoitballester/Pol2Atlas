module load homer/4.11
snakemake --snakefile source/07_homer_motif/homer.smk \
          --cores 54 \
          --rerun-incomplete \
          -k

