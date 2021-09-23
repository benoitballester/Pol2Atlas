# Using the Pol II catalog as an atlas for read counting in RNA-seq
We use as an example the TCGA RNA-seq atlas to identify key transcribed, non coding, intergenic, regulatory elements in cancer. Count tables are available to download here:

The code, the methodology and the atlas can be re-used for other RNA-seq datasets in BAM format. 
The script used to count reads for a single experiment is in "downloadCount/readCountsAtlas.sh". 
While our analyses alternatively download from the TCGA, count then delete the bam files, we also provide a snakemake pipeline to adapt our analyses to the general case where all files are available on disk.

The output files can be concatenated into an expression matrix with the "preprocessing/parseCounts.py" script.

