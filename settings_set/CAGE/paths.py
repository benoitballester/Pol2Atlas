# ---------- Common ---------- 
outputDir = "/shared/projects/pol2_chipseq/pol2_interg_default/output_CAGE/"   # Main results
tempDir = "/shared/projects/pol2_chipseq/pol2_interg_default/tempPol2/"       # Temporary files / unimportant
genomeFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/hg38.chrom.sizes.sorted"
annotationFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/peakDatasets/cage_data/cage_annot.csv"
polIIannotationPalette = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/palettes/annotation_palette3.tsv"
# ---------- Required for peak integrative analysis ---------- 
peaksFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/peakDatasets/cage_data/peaks/"
remapFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/remap2020_nr.bed"
GOfile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GO_files/hsapiens.GO:BP.name.gmt"
gencode = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/gencode.v38.annotation.gtf"
# ---------- Required for GWAS LDSC ---------- 
ldscFilesPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/ldsc/"
sumstatsFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data/GWAS/ldsc_sumstats/"
ldscSingularity = "/shared/projects/pol2_chipseq/pol2_interg_default/data/singularity/s-ldsc/envSingularityLDSC.img"
liftoverPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/liftover/"

# ---------- ROADMAP & Histones & Cons---------- 
roadmapPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/roadmap_chromatin_states/"
histonePathHeart27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/H3K27Ac_Chipseq/ENCFF962IUZ_heart_left_ventricule.bigWig"
histonePathLiver27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/H3K27Ac_Chipseq/ENCFF611FWS_liver.bigWig"
histonePathTcell27Ac = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/H3K27Ac_Chipseq/ENCFF750GKW_t-cell.bigWig"
histonePathHeart27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/Dnase/ENCFF169GLM_heart.bigWig"
histonePathLiver27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/Dnase/ENCFF556YQA_liver.bigWig"
histonePathTcell27Me3 = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/Dnase/ENCFF677BPI_Tcells.bigWig"
consFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/cons/hg38.phyloP100way.bw"
# ---------- Pol II specific ---------- 
# You will likely not need this for your analyses
bedtoolsPath = "/home/pdelangen/bedtools"
encodeBlacklist = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/hg38-blacklist.v2.bed"
# ---------- Genome annotation files ---------- 
ccrePath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/GRCh38-ccREsFix.bed"
lncpediaPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/lncipedia_5_2_hg38.bed"
repeatFamilyBed = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/repeats/repeatBedFamily.bed"
repeatClassBed ="/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/repeats/repeatBedClass.bed"
repeatTypeBed = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/repeats/repeatBedType.bed"
f5Enh ="/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/F5.hg38.enhancers.bed"
f5Cage = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/hg38_fair+new_CAGE_peaks_phase1and2.bed"
dnaseMeuleman = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/dnaseMeuleman.bed"
remapCrms = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/crm.bed"
# ---------- Required to download counts from TCGA ---------- 
tokenPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/tcga/token.txt"
manifest = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/gdc_manifest.2021-09-22.txt"
gdcClientPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/tcga/gdc-client"
tcgaData = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/tcga_metadata/"
# ---------- Required to download counts from GTEx ---------- 
gtexData = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/"
# ---------- Required only to read counts ---------- 
featureCountPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/tcga/featureCounts"
# ---------- Required for RNA-seq analysis ---------- 
countsTCGA = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/counts/"
tcgaGeneCounts = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/geneCounts.hd5"
tcgaAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/perFileAnnotation.tsv"
tcgaAnnotCounts = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/perFileAnnotationCounts.tsv"
tcgaToMainAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/tcga_project_annot.csv"
countsENCODE = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/encode_counts/"
encodeAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/annotation_files/encode_total_rnaseq_annot_0.tsv"
countsGTEx = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/gtex_counts/"
GTExAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/tsvs/sample.tsv"
atlasPath = "/shared/projects/pol2_chipseq/pol2_interg_default/temp_POL2_lenient_largerTSS_latestGENCODE/POLR2A_Inter_500.saf"
tableDirectory = outputDir + "rnaseq"
tissueToSimplified = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/annotation_files/tissue_to_simplified.csv"
# ---------- Required for scRNA-seq analysis ---------- 
pbmc10k = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/10k_pbmcs/"