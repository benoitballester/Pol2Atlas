outputDir = "/scratch/pdelangen/projet_these/outputPol2/"
tempDir = "/scratch/pdelangen/projet_these/tempPol2/"
peaksFolder = "/scratch/pdelangen/projet_these/data_clean/peaks/"
genomeFile = "/scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted"
annotationFile = "/scratch/pdelangen/projet_these/data_clean/annotPol2.tsv"
# Bed file with context name in 4th column and annotation priority 
# in 5th column (1 is lowest priority). If set to None does not perform
# context specific analyses
genomicContextBed = None