# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("./")
from settings import params, paths

classList = os.listdir(paths.tempDir)
classList = [i for i in classList if i.startswith("classifier")]
classNames = [i.split("-")[-1][:-4] for i in classList]
# %%
wAccuracies = []
for i in classList:
    with open(paths.tempDir + i) as f:
        l = f.readline()
        wAccuracies.append(float(l.split(" : ")[-1]))
wAccuracies = np.array(wAccuracies)
# %%
plt.figure(dpi=500)
order = np.argsort(wAccuracies)
plt.barh(np.arange(len(wAccuracies)),wAccuracies[order])
plt.yticks(np.arange(len(classNames)), np.array(classNames)[order])
plt.xticks(np.linspace(0,1.0,11))
plt.xlabel("Weighted accuracy")
plt.ylabel("Cancer type")
plt.vlines(0.5, plt.ylim()[0], plt.ylim()[1], color="red", linestyles="dashed")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(paths.outputDir + "rnaseq/global/predictive_results.png")
# %%
from lib.pyGREAT import pyGREAT
enricher = pyGREAT(oboFile=paths.GOfolder + "/go_eq.obo", geneFile=paths.gencode, 
                   geneGoFile=paths.GOfolder + "/goa_human.gaf")

# %%
import pyranges as pr
globalUpreg = pr.read_bed(paths.tempDir + "globallyDE_downreg.bed")
allConsensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
goEnrich = enricher.findEnriched(globalUpreg, allConsensuses)
from lib.utils import plot_utils
fig, ax = plt.subplots(figsize=(3,3),dpi=300)
plot_utils.enrichBarplot(ax, goEnrich["molecular_function"][1], 
                        goEnrich["molecular_function"][2], "GO Molecular Function enrichment", order_by="qval", fcMin=2.0)
plt.savefig(paths.outputDir + "rnaseq/global/go_mol_de_down.png", bbox_inches="tight")

# %%
import pyranges as pr
globalUpreg = pr.read_bed(paths.tempDir + "globallyDE_upreg.bed")
allConsensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
goEnrich = enricher.findEnriched(globalUpreg, allConsensuses)
from lib.utils import plot_utils
fig, ax = plt.subplots(figsize=(3,3),dpi=300)
plot_utils.enrichBarplot(ax, goEnrich["molecular_function"][1], 
                        goEnrich["molecular_function"][2], "GO Molecular Function enrichment", order_by="qval", fcMin=2.0)
plt.savefig(paths.outputDir + "rnaseq/global/go_mol_de_up.png", bbox_inches="tight")

# %%
import pyranges as pr
globalUpreg = pr.read_bed(paths.tempDir + "globallyProg.bed")
allConsensuses = pr.read_bed(paths.outputDir + "consensuses.bed")
goEnrich = enricher.findEnriched(globalUpreg, allConsensuses)
from lib.utils import plot_utils
fig, ax = plt.subplots(figsize=(3,3),dpi=300)
plot_utils.enrichBarplot(ax, goEnrich["molecular_function"][1], 
                        goEnrich["molecular_function"][2], "GO Molecular Function enrichment", order_by="qval", fcMin=2.0)
plt.savefig(paths.outputDir + "rnaseq/global/go_prog.png", bbox_inches="tight")

# %%
