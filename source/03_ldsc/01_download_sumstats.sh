snakemake --snakefile source/ldsc/dlGwasSumstats.smk \
          --cores 32 \
          --rerun-incomplete \
          -k \
          --cluster "sbatch -A b169 -p skylake -A b169 --time=8:00:0 -N 1 --ntasks-per-node=2 -o slurmOutput/job.out -e slurmOutput/job.err"

