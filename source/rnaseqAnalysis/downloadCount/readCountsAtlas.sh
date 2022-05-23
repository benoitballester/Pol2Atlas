# $1 : Annotation file
# $2 : Output count file
# $3 : Bam file to count
# $4 : Path to featureCount executable
# $5 : Temporary directory
# $6 : Thread count
# Create temp dir
mkdir $5
# Count reads using featureCount
# First attempt paired end reads
$4 -T 2 \
-F SAF \
-O \
-a $1 \
-p \
--countReadPairs \
-o $2 \
--tmpDir $5 \
$3 \
|| (\    # If it failed try single end
rm -rf $5 && mkdir $5 && \
$4 -T 2 \
-F SAF \
-O \
-a $1 \
-o $2 \
--tmpDir $5 \
$3)

                
    
# Cut last column, gunzip and erase original count file to save space
cut -f 7 $2 > $2.txt
gzip -f $2.txt
rm $2
rm -rf $5