# %%
import sys
import os
sys.path.append("./")
import numpy as np
import pandas as pd
import pyranges as pr
from lib.utils import overlap_utils


from settings import params, paths
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

intersectTable = pd.read_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t")
intersectTable.index = intersectTable["Name"]
# %%
# For globally DE probes
deBed = pr.read_bed(paths.outputDir + "rnaseq/TumorVsNormal/globally_DE.bed").as_df()
intersectTab = intersectTable.loc[deBed["Name"]]
try:
    os.mkdir(paths.outputDir + "intersections_databases/")
except:
    pass

proportions = intersectTab["Encode CCREs"].value_counts()/len(intersectTab)
plt.figure(dpi=500)
sns.barplot(proportions.values, proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/pancancer_DE_encode_ccres.pdf", bbox_inches="tight")
plt.show()
plt.close()

intersectLinc = intersectTab["lncpedia"].value_counts().sum()
intersectCCRE = intersectTab["Encode CCREs"].value_counts().sum()
intersectRemap = (intersectTab["ReMap CRM"] >= 10).sum()
intersectRepeat = intersectTab["Repeat Family"].value_counts().sum()
intersectF5Enh = intersectTab["Fantom 5 bidirectionnal enhancers"].sum()
intersectF5TSS = intersectTab["Fantom 5 TSSs"].sum()
intersectDnase = intersectTab["DNase meuleman"].value_counts().sum()
df = pd.DataFrame([intersectLinc, intersectCCRE, intersectRemap, intersectRepeat, intersectF5Enh,
                   intersectF5TSS, intersectDnase], index=["lncPedia", "Encode CCREs", "ReMap CRMs", "Repeated elements", "F5 enhancers", "F5 TSSs", "ENCODE Dnase"])
df /= len(intersectTab)
df.sort_values(0, inplace=True, ascending=False)
df["Cat"] = df.index
plt.figure(dpi=500)
sns.barplot(x=df[0], y=df.index)
plt.xticks(rotation=90)
plt.ylabel("Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/pancancer_DE_prop_all_db.pdf", bbox_inches="tight")
plt.show()
plt.close()

proportions = (intersectTab["Repeat Class"].value_counts())[:10]
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
N = len(intersectTable)
K = intersectTable["Repeat Class"].value_counts().loc[proportions.index]
n = len(intersectTab)
k = proportions
pvals = hypergeom(N,K,n).sf(k-1)
pvals = np.minimum(pvals, hypergeom(N,K,n).cdf(k-1))*2.0
sig = fdrcorrection(pvals)[0]

plt.figure(dpi=500)
sns.barplot(proportions.values/len(intersectTab), proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(10):
    fc = np.around((k/n)/(K/N), 2)[i]
    if sig[i]:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
plt.savefig(paths.outputDir + "intersections_databases/pancancer_DE_rep_family.pdf", bbox_inches="tight")
plt.show()
plt.close()
# %%
# For globally DE probes
intersectTab = pd.read_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t")
intersectTab.index = intersectTab["Name"]
progBed = pr.read_bed(paths.outputDir + "rnaseq/Survival/globally_prognostic.bed").as_df()
intersectTab = intersectTab.loc[progBed["Name"]]
try:
    os.mkdir(paths.outputDir + "intersections_databases/")
except:
    pass

proportions = intersectTab["Encode CCREs"].value_counts()/len(intersectTab)
plt.figure(dpi=500)
sns.barplot(proportions.values, proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/survival_encode_ccres.pdf", bbox_inches="tight")
plt.show()
plt.close()

intersectLinc = intersectTab["lncpedia"].value_counts().sum()
intersectCCRE = intersectTab["Encode CCREs"].value_counts().sum()
intersectRemap = (intersectTab["ReMap CRM"] >= 10).sum()
intersectRepeat = intersectTab["Repeat Family"].value_counts().sum()
intersectF5Enh = intersectTab["Fantom 5 bidirectionnal enhancers"].sum()
intersectF5TSS = intersectTab["Fantom 5 TSSs"].sum()
intersectDnase = intersectTab["DNase meuleman"].value_counts().sum()
df = pd.DataFrame([intersectLinc, intersectCCRE, intersectRemap, intersectRepeat, intersectF5Enh,
                   intersectF5TSS, intersectDnase], index=["lncPedia", "Encode CCREs", "ReMap CRMs", "Repeated elements", "F5 enhancers", "F5 TSSs", "ENCODE Dnase"])
df /= len(intersectTab)
df.sort_values(0, inplace=True, ascending=False)
df["Cat"] = df.index
plt.figure(dpi=500)
sns.barplot(x=df[0], y=df.index)
plt.xticks(rotation=90)
plt.ylabel("Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.savefig(paths.outputDir + "intersections_databases/survival_prop_all_db.pdf", bbox_inches="tight")
plt.show()
plt.close()

proportions = (intersectTab["Repeat Class"].value_counts())[:10]
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
N = len(intersectTable)
K = intersectTable["Repeat Class"].value_counts().loc[proportions.index]
n = len(intersectTab)
k = proportions
pvals = hypergeom(N,K,n).sf(k-1)
pvals = np.minimum(pvals, hypergeom(N,K,n).cdf(k-1))*2.0
sig = fdrcorrection(pvals)[0]

plt.figure(dpi=500)
sns.barplot(proportions.values/len(intersectTab), proportions.index)
plt.xticks(rotation=90)
plt.ylabel("CCRE Category")
plt.xlabel("Fraction of Pol II consensuses")
plt.xlim(0, plt.xlim()[1]*1.15)
for i in range(10):
    fc = np.around((k/n)/(K/N), 2)[i]
    if sig[i]:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc}", va="center")
    else:
        plt.text(proportions.values[i]/len(intersectTab) + plt.xlim()[1]*0.01, i, 
                 f"FC : {fc} (n.s)", va="center")
plt.savefig(paths.outputDir + "intersections_databases/survival_rep_family.pdf", bbox_inches="tight")
plt.show()
plt.close()
