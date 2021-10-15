# %%
# Retrieve sample state and patient survival
import numpy as np
import pandas as pd
from settings import params, paths

# %%
keys = pd.read_csv(paths.outputDir + "rnaseq/keys.txt", sep="\t", header=None)[0].values
manifest = pd.read_csv("/scratch/pdelangen/projet_these/data/tcga/BRCA_metadata/gdc_sample_sheet.2021-06-22.tsv", sep="\t", index_col=0)
annotation = np.array([manifest.loc[k]["Sample Type"] for k in keys])
cases = [manifest.loc[k]["Case ID"] for k in keys]
clinical = pd.read_csv("/scratch/pdelangen/projet_these/data/tcga/BRCA_metadata/clinical.tsv", 
                           sep="\t", index_col=1)
cancer_type = []
for i, c in enumerate(cases):
    if not annotation[i] == "Solid Tissue Normal":
        try:
            cancer_type.append(clinical.loc[c].iloc[0]["primary_diagnosis"])
        except:
            cancer_type.append("Unknown")
            print(c, "missing")
    else:
        cancer_type.append("Solid Tissue Normal")
survival = []
isDead = []
for i, c in enumerate(cases):
    try:
        data = clinical.loc[c].iloc[0]["days_to_death"]
        if data == "'--":
            isDead.append(0)
            survival.append(float(clinical.loc[c].iloc[0]["days_to_last_follow_up"]))
        else:
            survival.append(float(data))
            isDead.append(1)
    except:
        survival.append(-1)
        isDead.append(-1)
        print(c, "missing")
annotations = pd.DataFrame()
annotations["Sample"] = keys
annotations["State"] = annotation
annotations["Type"] = cancer_type
annotations["Dead"] = isDead
annotations["Time_to_event"] = survival
# %%
annotations.to_csv(paths.outputDir + "rnaseq/annotation.tsv", index=None, sep="\t")
# %%
