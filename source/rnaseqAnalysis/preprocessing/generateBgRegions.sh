./bedtools complement -i /scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed \
                    -g /scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted \
                    > /scratch/pdelangen/projet_these/data_clean/intergenicRegions_gc38.bed

./bedtools shuffle -excl /scratch/pdelangen/projet_these/outputPol2/consensuses.bed \
                 -incl /scratch/pdelangen/projet_these/data_clean/intergenicRegions_gc38.bed \
                 -maxTries 10000 \
                 -noOverlapping \
                 -seed 42 \
                 -i /scratch/pdelangen/projet_these/outputPol2/consensuses.bed \
                 -g /scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted \
                 > /scratch/pdelangen/projet_these/tempPol2/backgroundReg.bed