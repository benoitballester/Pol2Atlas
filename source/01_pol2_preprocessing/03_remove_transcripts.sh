# $1 = /scratch/pdelangen/projet_these/data_clean/filtered.bed
# $2 = /scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed
# $3 = /scratch/pdelangen/projet_these/data_clean/filteredInterg.bed

$1 subtract -a $2 \
            -b $3\
            -A \
            > $4

