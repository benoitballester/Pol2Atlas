snakemake --snakefile source/rnaseqAnalysis/downloadCount/GTex_counts.smk \
          --cores 42 \
          --rerun-incomplete \
          -k
#          --cluster "sbatch -A b169 -p kepler -A b169 --time=8:00:0 -N 1 --ntasks-per-node=2 -o slurmOutput/job.out -e slurmOutput/job.err"

