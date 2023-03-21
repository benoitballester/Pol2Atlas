dataFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data_all/"
outputDir = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2_repro/"   # Main results
tempDir = "/shared/projects/pol2_chipseq/pol2_interg_default/tempPol2_repro/"       # Temporary files / unimportant

# ---------- Common ---------- 
genomeFile = dataFolder + "hg38.chrom.sizes.sorted"
annotationFile = dataFolder + "dataset_annotation/annotPol2_lowlevel.tsv"
polIIannotationPalette = dataFolder + "palettes/annot.tsv"
singularityImg = dataFolder + "singularity/main/main_singularity.img"
# ---------- Pol II specific ---------- 
bedtoolsPath = dataFolder + "bin/bedtools"
encodeBlacklist = dataFolder + "genome_annotation/hg38-blacklist.v2.bed"
allPol2File = dataFolder + "pol2.bed"
allPol2Consensuses = dataFolder + "consensusesWholeGenome.bed"
# ---------- Required for peak integrative analysis ---------- 
peaksFolder = dataFolder + "Pol2_interg_peaks_per_dataset/"
remapFile = dataFolder + "genome_annotation/remap2020_nr.bed"
GOfile = dataFolder + "GO_files/hsapiens.GO:BP.name.gmt"
geneSets = dataFolder + "GO_files/"
gencode = dataFolder + "genome_annotation/gencode.v38.annotation.gtf"
# ---------- Required for GWAS LDSC ---------- 
ldscFilesPath = dataFolder + "ldsc/"
sumstatsFolder = dataFolder + "GWAS/ldsc_sumstats/"
ldscSingularity = dataFolder + "singularity/ldsc/ldsc_singularity.img"
liftoverPath = dataFolder + "liftover/"
# ---------- ROADMAP & Histones & Cons---------- 
roadmapPath = dataFolder + "roadmap_chromatin_states/"
histonePathHeart27Ac = dataFolder + "H3K27Ac_Chipseq/ENCFF962IUZ_heart_left_ventricule.bigWig"
histonePathLiver27Ac = dataFolder + "H3K27Ac_Chipseq/ENCFF611FWS_liver.bigWig"
histonePathTcell27Ac = dataFolder + "H3K27Ac_Chipseq/ENCFF750GKW_t-cell.bigWig"
histonePathHeart27Me3 = dataFolder + "Dnase/ENCFF169GLM_heart.bigWig"
histonePathLiver27Me3 = dataFolder + "Dnase/ENCFF556YQA_liver.bigWig"
histonePathTcell27Me3 = dataFolder + "Dnase/ENCFF677BPI_Tcells.bigWig"
consFile = dataFolder + "conservation/hg38.phyloP100way.bw"
# ---------- Genome annotation files ---------- 
ccrePath = dataFolder + "genome_annotation/GRCh38-ccREsFix.bed"
lncpediaPath = dataFolder + "genome_annotation/lncipedia_5_2_hg38.bed"
repeatFamilyBed = dataFolder + "genome_annotation/repeatBedFamily.bed"
repeatClassBed =dataFolder + "genome_annotation/repeatBedClass.bed"
repeatTypeBed = dataFolder + "genome_annotation/repeatBedType.bed"
f5Enh =dataFolder + "genome_annotation/F5.hg38.enhancers.bed"
f5Cage = dataFolder + "genome_annotation/hg38_fair+new_CAGE_peaks_phase1and2.bed"
dnaseMeuleman = dataFolder + "genome_annotation/dnaseMeuleman.bed"
remapCrms = dataFolder + "genome_annotation/crm.bed"
# ---------- Required to download counts from TCGA ---------- 
tokenPath = dataFolder + "tcga/token.txt" # Use your own !
manifest = dataFolder + "gdc_manifest.2021-09-22.txt"
gdcClientPath = dataFolder + "tcga/gdc-client"
tcgaData = dataFolder + "tcga/"
# ---------- Required to download counts from GTEx ---------- 
gtexData = dataFolder + "GTex/"
# ---------- Required only to read counts ---------- 
featureCountPath = dataFolder + "bin/featureCounts"
# ---------- Required for RNA-seq analysis ---------- 
countsTCGA = dataFolder + "rnaseq_counts_per_file/tcga/"
countsENCODE = dataFolder + "rnaseq_counts_per_file/encode/"
countsGTEx = dataFolder + "rnaseq_counts_per_file/gtex/"
atlasPath = dataFolder + "rnaseq_counts_per_file/saf_files/POLR2A_Inter_500.saf"
encodeMetadata = dataFolder + "encode_metadata_totalrnaseq/metadata_encode_totalrnaseq.tsv"
tcgaGeneCounts = dataFolder + "rnaseq_gene_counts/geneCounts.hd5"
tcgaAnnot = dataFolder + "tcga/perFileAnnotation.tsv"
tcgaSubtypes = dataFolder + "tcga/molecular_subtypes.csv"
tcgaAnnotCounts = dataFolder + "tcga/perFileAnnotationCounts.tsv"
tcgaToMainAnnot = dataFolder + "tcga/tcga_project_annot.csv"
encodeAnnot = dataFolder + "dataset_annotation/encode_total_rnaseq_annot_0.tsv"
GTExAnnot = dataFolder + "GTex/tsvs/sample_annot.tsv"
gtexGeneCounts = dataFolder + "rnaseq_gene_counts/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct"
tissueToSimplified = dataFolder + "dataset_annotation/tissue_to_simplified.csv"
ensIdToGeneId = dataFolder + "/ensembl_toGeneId.tsv"
hallmarksGenes = dataFolder + "/cosmic_hallmarks/hallmarks_chg.csv"