# Arguments
# $1 : Genome File (/scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted)
# $2 : Genic regions file (+extent) (/scratch/pdelangen/projet_these/data_clean/genicRegions_gc38.bed)
# $3 : Output Directory (/scratch/pdelangen/projet_these/tempPol2)
# $4 : Pol II file (/scratch/pdelangen/projet_these/outputPol2/consensuses.bed)
# Generate intergenic regions bed
/home/pdelangen/bedtools complement -i $2 \
                    -g $1 \
                    > $3/intergenicRegions_gc38.bed

# Generate SAFs file from bed
source/rnaseqAnalysis/downloadCount/generate_SAFs.py $4 Pol2_Intergenic

# Shuffle
/home/pdelangen/bedtools shuffle -excl $3/Pol2_Intergenic_500.bed \
                                -incl $3/intergenicRegions_gc38.bed \
                                -maxTries 10000 \
                                -noOverlapping \
                                -seed 42 \
                                -chrom \
                                -i $3/Pol2_Intergenic_500.bed \
                                -g $1 \
                                > $3/backgroundReg.bed

# Convert shuffled beds to SAF
python source/rnaseqAnalysis/downloadCount/bed_to_saf.py /scratch/pdelangen/projet_these/tempPol2/backgroundReg.bed
