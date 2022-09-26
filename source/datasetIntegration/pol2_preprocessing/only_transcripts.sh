bedtools intersect -a /shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filtered.bed \
                  -b /shared/projects/pol2_chipseq/pol2_interg_default/data_clean/genicRegions_gc38.bed -f 1.0 -u \
                   > /shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredGenic.bed
