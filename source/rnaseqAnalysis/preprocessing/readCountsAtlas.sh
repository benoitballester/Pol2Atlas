# Count reads using feature counts (single end sequencing)
./featureCounts -T 1 -F SAF -O \
    --largestOverlap -a {parameters.tempDir}POLR2A_Inter_500.saf \
    -p -o {countDir500}/{wildcards.fileIDs}.counts \
    {parameters.dataPath}tcga/counts/temp/{wildcards.fileIDs}/{params.nameID}
    
# Cut last column, gunzip and erase original count file to save space
cut -f 7 {countDir500}/{wildcards.fileIDs}.counts > {countDir500}/{wildcards.fileIDs}.txt
gzip {countDir500}/{wildcards.fileIDs}.txt
rm {countDir500}/{wildcards.fileIDs}.counts 