# ---------- Common ---------- 
outputDir = "/scratch/pdelangen/projet_these/outputPol2/"   # Main results
tempDir = "/scratch/pdelangen/projet_these/tempPol2/"       # Mostly temporary/unimportant
genomeFile = "/scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted"
annotationFile = "/scratch/pdelangen/projet_these/data_clean/annotPol2.tsv"

# ---------- Required for peak integrative analysis ---------- 
peaksFolder = "/scratch/pdelangen/projet_these/data_clean/peaksInterg/"
remapFile = "/scratch/pdelangen/projet_these/data_clean/remap2020_nr.bed"
repeatFile = "/scratch/pdelangen//projet_these/oldBackup/temp_POL2_lenient/repeatBedType.bed"
GOfolder = "/scratch/pdelangen/projet_these/data_clean/GO_annotations"
gencode = "/scratch/pdelangen/projet_these/data/annotation/gencode.v38.annotation.gtf"
# ---------- Required for GWAS LDSC ---------- 
ldscFilesPath = "/scratch/pdelangen/projet_these/data_clean/ldsc/"
ldscSingularity = "/scratch/pdelangen/projet_these/data/singularity/s-ldsc/envSingularityLDSC.img"
liftoverPath = "/scratch/pdelangen/projet_these/data/liftover/"
# ---------- Pol II specific ---------- 
# You will likely not need this for your analyses
bedtoolsPath = "/home/pdelangen/bedtools"
encodeBlacklist = "/scratch/pdelangen/projet_these/data_clean/hg38-blacklist.v2.bed"
# ---------- Required to download counts from TCGA ---------- 
tokenPath = "/scratch/pdelangen/projet_these/data/tcga/token.txt"
manifest = "/scratch/pdelangen/projet_these/data_clean/gdc_manifest.2021-09-22.txt"
gdcClientPath = "/scratch/pdelangen/projet_these/data/tcga/gdc-client"

# ---------- Required only to read counts ---------- 
featureCountPath = "/scratch/pdelangen/projet_these/data/tcga/featureCounts"

# ---------- Required for RNA-seq analysis ---------- 
countDirectory = "/scratch/pdelangen/projet_these/outputPol2/rnaseq/counts/"
atlasPath = "/scratch/pdelangen/projet_these/temp_POL2_lenient_largerTSS_latestGENCODE/POLR2A_Inter_500.saf"
tableDirectory = outputDir + "rnaseq"