options(repos=structure(c(CRAN="https://pbil.univ-lyon1.fr/CRAN/")))
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", dependencies=TRUE)

# Limma
BiocManager::install("edgeR")

# Scran
BiocManager::install("scran")

# Deseq2
BiocManager::install("DESeq2")