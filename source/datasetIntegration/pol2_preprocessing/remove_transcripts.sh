# $1 = /scratch/pdelangen/projet_these/data_clean/filtered.bed
# $2 = /scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed
# $3 = /scratch/pdelangen/projet_these/data_clean/filteredInterg.bed

bedtools subtract -a "$1" \
                  -b "$2" -A \
                   > "$3"
