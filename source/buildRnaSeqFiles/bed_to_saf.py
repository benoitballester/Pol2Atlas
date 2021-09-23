# %%
import pandas as pd
import sys
sys.path.append("./")
from settings import params, paths
import subprocess

# sys.argv[1] = "/scratch/pdelangen/projet_these/tempPol2/backgroundReg.bed"
f = pd.read_csv(sys.argv[1], sep="\t", header=None)
f.columns = ["Chr", "Start", "End"]
f.index.name = "GeneID"
f.to_csv(sys.argv[1].split(".")[0] + ".saf", sep="\t")
# %%
