# %%
import sys
sys.path.append("./")
from settings import params, paths
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from lib.utils import utils
utils.createDir(paths.outputDir + "homer_motifs/all_figures/")
folders = os.listdir(paths.outputDir + "homer_motifs/")
# %%
for f in folders:
    if 'knownResults.txt' in os.listdir(paths.outputDir + "homer_motifs/" + f):
        tab = pd.read_csv(paths.outputDir + "homer_motifs/" + f + '/knownResults.txt',sep="\t")
        tab.index = tab["Motif Name"].str.split("/", expand=True)[0]
        tab.index = pd.Series(tab.index).str.split("(", expand=True)[0]
        res = -tab["Log P-value"][:8]
        plt.figure(dpi=500)
        sns.barplot(y=res.index, x=res.values)
        plt.xlabel("-log(p-value)")
        plt.ylabel("Motif")
        plt.gca().set_aspect(np.ptp(plt.xlim())/np.ptp(plt.ylim()))
        plt.savefig(paths.outputDir + "homer_motifs/all_figures/" + f + ".pdf", bbox_inches="tight")
        plt.close()
        
# %%
