# ---------- Common ---------- 
outputDir = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/"   # Main results
tempDir = "/shared/projects/pol2_chipseq/pol2_interg_default/tempPol2/"       # Temporary files / unimportant
genomeFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/hg38.chrom.sizes.sorted"
annotationFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/annotPol2_cpy.tsv"
polIIannotationPalette = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/palettes/annotation_palette2.tsv"
# ---------- Required for peak integrative analysis ---------- 
peaksFolder = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/peaksInterg/"
remapFile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/remap2020_nr.bed"
repeatFile = "/shared/projects/pol2_chipseq/pol2_interg_default/oldBackup/temp_POL2_lenient/repeatBedType.bed"
GOfile = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GO_files/hsapiens.GO:BP.name.gmt"
gencode = "/shared/projects/pol2_chipseq/pol2_interg_default/data/annotation/gencode.v38.annotation.gtf"
# ---------- Required for GWAS LDSC ---------- 
ldscFilesPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/ldsc/"
ldscSingularity = "/shared/projects/pol2_chipseq/pol2_interg_default/data/singularity/s-ldsc/envSingularityLDSC.img"
liftoverPath = "/shared/projects/pol2_chipseq/pol2_interg_default/data/liftover/"
# ---------- Pol II specific ---------- 
# You will likely not need this for your analyses
bedtoolsPath = "/home/pdelangen/bedtools"
encodeBlacklist = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/hg38-blacklist.v2.bed"



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
countsENCODE = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/encode_counts/"
encodeAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/encode_total_rnaseq_annot_0 (copy).tsv"
countsGTEx = "/shared/projects/pol2_chipseq/pol2_interg_default/outputPol2/rnaseq/gtex_counts/"
GTExAnnot = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/tsvs/sample.tsv"
atlasPath = "/shared/projects/pol2_chipseq/pol2_interg_default/temp_POL2_lenient_largerTSS_latestGENCODE/POLR2A_Inter_500.saf"
tableDirectory = outputDir + "rnaseq"