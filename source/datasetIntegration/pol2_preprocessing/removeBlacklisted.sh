# $1 = /scratch/pdelangen/projet_these/data_clean/pol2.bed
# $2 = /scratch/pdelangen/projet_these/data_clean/hg38-blacklist.v2.bed
# $3 = /scratch/pdelangen/projet_these/data_clean/filtered.bed

bedtools subtract -a "$1" \
                  -b "$2"\
                  -A \
                   > "$3"
