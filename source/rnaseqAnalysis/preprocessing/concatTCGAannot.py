# %%
import pandas as pd
import numpy as np
import os

# Merge annot of bam files
clinical1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch1/clinical.tsv", sep="\t")
clinical2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch2/clinical.tsv", sep="\t")
clinical = pd.concat([clinical1,clinical2])
clinical = clinical.drop_duplicates("case_submitter_id")
# %%
samples1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch1/samples.tsv", sep="\t")
samples2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch2/samples.tsv", sep="\t")
samples = pd.concat([samples1,samples2])
# %%
aliquots1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch1/aliquot.tsv", sep="\t")
aliquots2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/batch2/aliquot.tsv", sep="\t")
aliquots = pd.concat([aliquots1,aliquots2])
# %%
merged = clinical.merge(samples, right_on="Case ID", left_on="case_submitter_id")
merged.index = merged["File ID"]
# %%
merged2 = merged.merge(aliquots, left_on="Sample ID", right_on="sample_submitter_id")
# %%
merged.to_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotation.tsv", sep="\t")
# %%
# Merge annot of counts file
clinical1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countTablesTCGA1/clinical.tsv", sep="\t")
clinical2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countsTablesTCGA2/clinical.tsv", sep="\t")
clinical = pd.concat([clinical1,clinical2])
clinical = clinical.drop_duplicates("case_submitter_id")
# %%
samples1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countTablesTCGA1/samples.tsv", sep="\t")
samples2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countsTablesTCGA2/samples.tsv", sep="\t")
samples = pd.concat([samples1,samples2])
# %%
aliquots1 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countTablesTCGA1/aliquot.tsv", sep="\t")
aliquots2 = pd.read_csv("/scratch/pdelangen/projet_these/data_clean/countsTablesTCGA2/aliquot.tsv", sep="\t")
aliquots = pd.concat([aliquots1,aliquots2])
# %%
merged = clinical.merge(samples, right_on="Case ID", left_on="case_submitter_id")
merged.index = merged["File ID"]
# %%
# merged2 = merged.merge(aliquots, left_on="sample_id", right_on="sample_submitter_id")
# %%
merged.to_csv("/scratch/pdelangen/projet_these/data_clean/perFileAnnotationCounts.tsv", sep="\t")
# %%
# Concatenate count table
tablePath1 = "/scratch/pdelangen/projet_these/data_clean/countTablesTCGA1/"
tablePath2 = "/scratch/pdelangen/projet_these/data_clean/countsTablesTCGA2/"
tableList1 = os.listdir(tablePath1)
allCounts = []
ids = []
for f in tableList1:
    if os.path.isdir(tablePath1 + f):
        f2 = os.listdir(tablePath1 + f)
        f2 = [i for i in f2 if i.endswith(".gz")][0]
        allCounts.append(pd.read_csv(tablePath1 + f  + "/" + f2, sep="\t", 
                                    header=None, index_col=0, squeeze=True, dtype={0:str,1:"int32"}))
        ids.append(f)
tableList2 = os.listdir(tablePath2)
for f in tableList2:
    if os.path.isdir(tablePath2 + f):
        f2 = os.listdir(tablePath2 + f)
        f2 = [i for i in f2 if i.endswith(".gz")][0]
        allCounts.append(pd.read_csv(tablePath2 + f  + "/" + f2, sep="\t", 
                                    header=None, index_col=0, squeeze=True, dtype={0:str,1:"int32"}))
        ids.append(f)




# %%
countsConcat = pd.concat(allCounts, axis=1)
# %%
countsConcat.columns = ids
# %%
countsConcat.to_hdf("/scratch/pdelangen/projet_these/data_clean/geneCounts.hd5", key="Counts")
# %%
test = pd.read_hdf("/scratch/pdelangen/projet_these/data_clean/geneCounts.hd5")
# %%
