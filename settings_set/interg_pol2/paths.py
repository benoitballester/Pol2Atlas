# ---------- Common ---------- 
outputDir = "/scratch/pdelangen/projet_these/outputPol2_Interg/"   # Main results
tempDir = "/scratch/pdelangen/projet_these/tempPol2_Interg/"       # Mostly temporary/unimportant
genomeFile = "/scratch/pdelangen/projet_these/data_clean/hg38.chrom.sizes.sorted"
annotationFile = "/scratch/pdelangen/projet_these/data_clean/annotPol2_All.tsv"

# ---------- Required for peak integrative analysis ---------- 
peaksFolder = "/scratch/pdelangen/projet_these/data_clean/peaksInterg/"

# ---------- Pol II specific ---------- 
# You will likely not need this for your analyses
gencode = "/scratch/pdelangen/projet_these/data/annotation/gencode.v38.annotation.gtf"
bedtoolsPath = "/home/pdelangen/bedtools"
encodeBlacklist = "/scratch/pdelangen/projet_these/data_clean/hg38-blacklist.v2.bed"
# ---------- Required for GWAS analysis ---------- 


# ---------- Required to download counts from TCGA ---------- 
tokenPath = "/scratch/pdelangen/projet_these/data/tcga/token.txt"
manifest = "/scratch/pdelangen/projet_these/data_clean/gdc_manifest.2021-09-22.txt"
gdcClientPath = "/scratch/pdelangen/projet_these/data/tcga/gdc-client"

# ---------- Required only to read counts ---------- 
featureCountPath = "/scratch/pdelangen/projet_these/data/tcga/featureCounts"

# ---------- Required for RNA-seq analysis ---------- 
countDirectory = "/scratch/pdelangen/projet_these/output_POL2_lenient_largerTSS_latestGENCODE/counts/500centroid/"
atlasPath = "/scratch/pdelangen/projet_these/temp_POL2_lenient_largerTSS_latestGENCODE/POLR2A_Inter_500.saf"
tableDirectory = outputDir + "rnaseq"