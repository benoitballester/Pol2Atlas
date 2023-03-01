# $1 = /scratch/pdelangen/projet_these/data_clean/pol2.bed
# $2 = /scratch/pdelangen/projet_these/data_clean/hg38-blacklist.v2.bed
# $3 = /scratch/pdelangen/projet_these/data_clean/filtered.bed
# $4 bedtools

$1 subtract -a $2 \
            -b $3\
            -A \
            > $4
