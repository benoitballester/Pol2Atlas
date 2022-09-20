./cellranger mkref \
            --genome=hsap \
            --fasta=/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
            --genes=/shared/projects/pol2_chipseq/pol2_interg_default/tempPol2/Pol2.gtf
./cellranger count --id=10k_PBMC \
            --transcriptome=hsap \
            --fastqs=/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/10k_pbmcs/10k_PBMC_3p_nextgem_Chromium_X_fastqs/ \
            --sample=10k_PBMC_3p_nextgem_Chromium_X \
            --expect-cells=10000
