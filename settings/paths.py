# ---------- Common ---------- 
outputDir = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2_Annot3/"   # Main results
tempDir = "/shared/projects/pol2_chipseq/pol2_interg_default/tempPol2/"       # Temporary files / unimportant
genomeFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/hg38.chrom.sizes.sorted"
annotationFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/dataset_annotation/annotPol2_lowlevel.tsv"
polIIannotationPalette = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/palettes/annot.tsv"
# ---------- Required for peak integrative analysis ---------- 
peaksFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/Pol2_interg_peaks_per_dataset/"
remapFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/remap2020_nr.bed"
GOfile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/GO_files/hsapiens.GO:BP.name.gmt"
gencode = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/gencode.v38.annotation.gtf"
# ---------- Required for GWAS LDSC ---------- 
ldscFilesPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/ldsc/"
sumstatsFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data/GWAS/ldsc_sumstats/"
ldscSingularity = "/shared/projects/pol2_chipseq/pol2_interg_default/data/singularity/s-ldsc/envSingularityLDSC.img"
liftoverPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/liftover/"
# ---------- ROADMAP & Histones & Cons---------- 
roadmapPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/roadmap_chromatin_states/"
histonePathHeart27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/H3K27Ac_Chipseq/ENCFF962IUZ_heart_left_ventricule.bigWig"
histonePathLiver27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/H3K27Ac_Chipseq/ENCFF611FWS_liver.bigWig"
histonePathTcell27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/H3K27Ac_Chipseq/ENCFF750GKW_t-cell.bigWig"
histonePathHeart27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/Dnase/ENCFF169GLM_heart.bigWig"
histonePathLiver27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/Dnase/ENCFF556YQA_liver.bigWig"
histonePathTcell27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/Dnase/ENCFF677BPI_Tcells.bigWig"
consFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/conservation/hg38.phyloP100way.bw"
# ---------- Pol II specific ---------- 
# You will likely not need this for your analyses
bedtoolsPath = "/home/pdelangen/bedtools"
encodeBlacklist = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/hg38-blacklist.v2.bed"
allPol2File = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/pol2.bed"
# ---------- Genome annotation files ---------- 
ccrePath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/GRCh38-ccREsFix.bed"
lncpediaPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/lncipedia_5_2_hg38.bed"
repeatFamilyBed = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/repeatBedFamily.bed"
repeatClassBed ="/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/repeatBedClass.bed"
repeatTypeBed = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/repeatBedType.bed"
f5Enh ="/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/F5.hg38.enhancers.bed"
f5Cage = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/hg38_fair+new_CAGE_peaks_phase1and2.bed"
dnaseMeuleman = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/dnaseMeuleman.bed"
remapCrms = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/genome_annotation/crm.bed"
# ---------- Required to download counts from TCGA ---------- 
tokenPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga/token.txt" # Use your own !
manifest = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/gdc_manifest.2021-09-22.txt"
gdcClientPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga/gdc-client"
tcgaData = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga_metadata/"
# ---------- Required to download counts from GTEx ---------- 
gtexData = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/GTex/"
# ---------- Required only to read counts ---------- 
featureCountPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/bin/featureCounts"
# ---------- Required for RNA-seq analysis ---------- 
countsTCGA = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq_counts_per_file/tcga/"
countsENCODE = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq_counts_per_file/encode/"
countsGTEx = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq_counts_per_file/gtex/"
tcgaGeneCounts = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/rnaseq_gene_counts/geneCounts.hd5"
tcgaAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga/perFileAnnotation.tsv"
tcgaAnnotCounts = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga/perFileAnnotationCounts.tsv"
tcgaToMainAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/tcga_project_annot.csv"
encodeAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/dataset_annotation/encode_total_rnaseq_annot_0.tsv"
GTExAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/GTex/tsvs/sample.tsv"
atlasPath = "/shared/projects/pol2_chipseq/pol2_interg_default/temp_POL2_lenient_largerTSS_latestGENCODE/POLR2A_Inter_500.saf"
tableDirectory = outputDir + "rnaseq"
tissueToSimplified = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/dataset_annotation/tissue_to_simplified.csv"
# ---------- Required for scRNA-seq analysis ---------- 
pbmc10k = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/10k_pbmcs/"