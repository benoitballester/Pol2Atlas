# $1 : Annotation file
# $2 : Output count file
# $3 : Bam file to count
# $4 : Path to featureCount executable
# Count reads using feature counts
# First attempt paired end reads
$4 -T 1 \
-F SAF \
-O \
-a $1 \
-p \
--countReadPairs \
-o $2 \
$3 \
|| \    # If it failed try single end
$4 -T 1 \
-F SAF \
-O \
-a $1 \
-o $2 \
$3

                
    
# Cut last column, gunzip and erase original count file to save space
cut -f 7 $2 > $2.txt
gzip -f $2.txt
rm $2