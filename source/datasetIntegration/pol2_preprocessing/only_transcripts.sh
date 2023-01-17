# $1 = /scratch/pdelangen/projet_these/data_clean/filtered.bed
# $2 = /scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed
# $3 = /shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredGenic.bed


bedtools intersect -a "$1" \
                  -b "$2" -f 1.0 -u \
                   > "$3"
