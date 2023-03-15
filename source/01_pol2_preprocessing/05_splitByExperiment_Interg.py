# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("./")
from settings import params, paths
from lib.utils import utils
utils.createDir(paths.outputDir + "descriptivePlots/")

""" sys.argv = [None]*6
sys.argv[1] = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredInterg.bed"
sys.argv[2] = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/peaks/"
sys.argv[3] = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filtered.bed"
sys.argv[4] = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredqval_dataset_all.bed"
sys.argv[5] = "/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/filteredqval_dataset_interg.bed" """
print(sys.argv)
df = pd.read_csv(f"{sys.argv[1]}", 
                 sep="\t", header=None)
logQvalCutoff = params.logQvalCutoff
peakCountCutoff = params.peakCountCutoff
# %%
hasPol2 = df[3].str.split(".", expand=True)[1] == "POLR2A"
# %%
selected = df[hasPol2]
plt.figure()
plt.hist(selected[4], np.arange(0,100))
plt.vlines(logQvalCutoff, plt.ylim()[0], plt.ylim()[1], "red", "dashed")
plt.xlabel("-log10(q-value) per peak")
plt.ylabel("Number of peaks")
plt.savefig(paths.outputDir + "descriptivePlots/distribQval.pdf")
plt.show()
selected = selected[selected[4] > logQvalCutoff]
selected = selected[selected[0] != "chrEBV"]
# %%
dup = selected
dup[[9,10]] = selected[3].str.split("_peak", expand=True) 
dup
# %%
splited = dict(([(k,x[np.arange(8)]) for k, x in dup.groupby(9)]))
# %%
peaksPerDataset = []
keptDatasets = []
for f in splited.keys():
    if len(splited[f]) > peakCountCutoff:
        splited[f].to_csv(f"{sys.argv[2]}{f}.bed.gz", 
                        sep="\t", header=False, index=False)
        keptDatasets.append(f)
    peaksPerDataset.append(len(splited[f]))
peaksPerDataset = np.array(peaksPerDataset)
# %%
plt.figure()
plt.hist(np.sort(peaksPerDataset), np.logspace(0, 5.2,50))
plt.xscale("log")
plt.vlines(peakCountCutoff, 0, 100, "red", "dashed")
plt.xlabel("Intergenic peak count per dataset")
plt.ylabel("Number of datasets")
plt.savefig(paths.outputDir + "descriptivePlots/distribIntergPeaksPerDataset.pdf")
plt.show()

# %%
palette = pd.read_csv(paths.polIIannotationPalette, sep=",", index_col="Annotation")
paletteStr = [",".join((c[1].values*255).astype(int).astype(str)) for c in palette.iterrows()]
paletteStr = dict(zip(palette.index, paletteStr))
annot = dict(pd.read_csv(paths.annotationFile, sep="\t", index_col="Sample")["Annotation"])
keptDatasets = set(keptDatasets)

# %%
keptDatasets = set(keptDatasets)
newFile = sys.argv[5]
with open(newFile, "w") as f:
    i = 0
    for l in selected.values[:,:9]:
        dataset = l[3].split("_peak")[0]
        if dataset in keptDatasets:               
            biotype = annot[dataset + ".bed.gz"]
            l[8] = paletteStr[biotype]
            print("\t".join(l.astype(str)), file=f)

# %%
dfFull = pd.read_csv(f"{sys.argv[3]}", 
                 sep="\t", header=None)
hasPol2 = dfFull[3].str.split(".", expand=True)[1] == "POLR2A"
# %%
selected = dfFull[hasPol2]
selected = selected[selected[4] > logQvalCutoff]
selected = selected[selected[0] != "chrEBV"]
# %%
keptDatasets = set(keptDatasets)
newFile = sys.argv[4]
with open(newFile, "w") as f:
    i = 0
    for l in selected.values:
        dataset = l[3].split("_peak")[0]
        if dataset in keptDatasets:               
            biotype = annot[dataset + ".bed.gz"]
            l[8] = paletteStr[biotype]
            print("\t".join(l.astype(str)), file=f)
        
        
# %%
