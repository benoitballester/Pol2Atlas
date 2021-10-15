# %%
import pandas as pd
import numpy as np
import os


clinical1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch1/clinical.tsv", sep="\t")
clinical2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch2/clinical.tsv", sep="\t")
clinical = pd.concat([clinical1,clinical2])
clinical = clinical.drop_duplicates("case_submitter_id")
# %%
samples1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch1/samples.tsv", sep="\t")
samples2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch2/samples.tsv", sep="\t")
samples = pd.concat([samples1,samples2])
# %%
merged = clinical.merge(samples, right_on="Case ID", left_on="case_submitter_id")
merged.index = merged["File ID"]
# %%
merged.to_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", sep="\t")
# %%
