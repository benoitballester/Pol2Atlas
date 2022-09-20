# %%
import pyranges as pr
import numpy as np
import pandas
import os
import urllib.request
import sys
sys.path.append("./")
from settings import params, paths

clusterPath = paths.tempDir + "liftedClusters/"
files = os.listdir(clusterPath)
bedfiles = [pr.read_bed(clusterPath + c) for c in files]
# %%
# First check if TF intersects with any HapMap3 SNPs
hasIntersect = np.zeros(len(files), dtype=int)
for ch in range(1,23):
    snps = pandas.read_csv(paths.ldscFilesPath + f"ldsc_files/1000G_EUR_Phase3_plink/1000G.EUR.QC.{ch}.bim", sep="\t", header=None)
    snps.columns = ["Chromosome", "Name", "cm", "Start", "a1", "a2"]
    snpAnnotFmt = snps[["Chromosome", "Start", "Name", "cm"]]
    snpAnnotFmt.columns = ["CHR", "BP", "SNP", "CM"]
    snpAnnotFmt.index = snpAnnotFmt["SNP"]
    snps["End"] = snps["Start"] + 1
    snpBed = snps[["Chromosome", "Start", "End", "Name"]]
    snpBed["Chromosome"] = "chr" + snpBed["Chromosome"].astype("string")
    snpPr = pr.PyRanges(df=snpBed)
    for i, tf in enumerate(files):
        print(tf, ch)
        intersection = snpPr.overlap(bedfiles[i], how=None).as_df()
        hasIntersect[i] += len(intersection)


# %%
for ch in range(1,23):
    snps = pandas.read_csv(paths.ldscFilesPath + f"ldsc_files/1000G_EUR_Phase3_plink/1000G.EUR.QC.{ch}.bim", sep="\t", header=None)
    snps.columns = ["Chromosome", "Name", "cm", "Start", "a1", "a2"]
    snpAnnotFmt = snps[["Chromosome", "Start", "Name", "cm"]]
    snpAnnotFmt.columns = ["CHR", "BP", "SNP", "CM"]
    snpAnnotFmt.index = snpAnnotFmt["SNP"]
    snps["End"] = snps["Start"]+1
    snpBed = snps[["Chromosome", "Start", "End", "Name"]]
    snpBed["Chromosome"] = "chr" + snpBed["Chromosome"].astype("string")
    snpPr = pr.PyRanges(df=snpBed)
    for i, f in enumerate(files):
        if hasIntersect[i] > 100:   # Remove groups with < 100 SNPs
            print(f, ch)
            annot = clusterPath + f
            annotName = annot.split(".")[0]
            snpAnnotFmt[annotName] = 0
            snpAnnotFmt[annotName].astype('int8')
            try:
                intersection = snpPr.overlap(bedfiles[i], how=None).as_df()["Name"]
                snpAnnotFmt[annotName].loc[intersection] = 1
            except KeyError:
                pass
    snpAnnotFmt.to_csv(paths.tempDir + f"ld.{ch}.annot.gz", index=False, sep=" ")
# %%
