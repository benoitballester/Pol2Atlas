library(edgeR)
library(Matrix)
# Read command line arguments
args = commandArgs(trailingOnly=TRUE)
print(args)
countsPath = args[1]
annotationPath = args[2]
probeNames = args[3]
sfPath = args[4]
DEFolder = args[5]
# Load matrix, labels, size factors, Pol II probe id
ann = read.delim(annotationPath,sep=",")
groups= as.factor(gsub(" ", "_",ann[, 2]))
counts <- t(readMM(countsPath))
colnames(counts) = 1:ncol(counts)
rownames(counts) = c(read.delim(probeNames, header=FALSE))$V1
d0 <- DGEList(counts)
# Squeezing in our custom size factors
sf = c(read.delim(sfPath, header=FALSE))$V1
d0$samples$norm.factors = sf
# Design matrix
mm <- model.matrix(~0 + groups)
print(mm)
rm(counts)
gc()
# Fit voom
y <- voom(d0, mm, plot = T)
# Free some memory
rm(d0)
gc()
# Fit lm
fit <- lmFit(y, mm)
rm(y)
gc()
# Compute one vs all DE for each label
for (i in 1:ncol(mm))
{
  currGroup = colnames(mm)[i]
  fml = paste(as.character(currGroup)," - (", sep="")
  for (j in 1:ncol(mm))
  {
    if (i != j & j != ncol(mm)) {
      fml = paste(fml, as.character(colnames(mm)[j])," + ", sep="")
    }
    if (j == ncol(mm)) {
      fml = paste(fml, as.character(colnames(mm)[j]), ")/",as.character(j-1), sep="")
    }
  }
  contr <- makeContrasts(fml, levels = colnames(coef(fit)))
  tmp <- contrasts.fit(fit, contr)
  tmp <- eBayes(tmp)
  top.table <- topTable(tmp, sort.by = "P", n = Inf)
  write.csv(top.table, paste(DEFolder, currGroup, ".csv"))
  rm(contr)
  rm(tmp)
  gc()
}
# Null residuals
# residualSave = args[6]
# mm <- model.matrix(~0+rep(1, ncol(counts)))
# y <- voom(d0, mm, plot = T)
# fit <- lmFit(y, mm)
# r = residuals(fit, y)
# write.csv(r, gzfile(paste(residualSave, "nullResiduals.csv.gz")))