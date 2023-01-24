snakemake --snakefile source/rnaseqAnalysis/downloadCount/GTex_counts.smk \
          --cores 40 \
          --rerun-incomplete \
          -k
          # --cluster "sbatch -A pol2_chipseq -p fast --time=18:00:0 -N 1 --ntasks-per-node=2 -o slurmOutput/job.out -e slurmOutput/job.err"

