snakemake --snakefile source/rnaseqAnalysis/downloadCount/chagas_rnaseq_counts.smk \
          --jobs 1 \
          --rerun-incomplete \
          -k --cluster "sbatch -A pol2_chipseq -p fast --time=8:00:0 --cpus-per-task 24 -o slurmOutput/job.out -e slurmOutput/job.err"
