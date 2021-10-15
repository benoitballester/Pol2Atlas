# %%
# Parse xml sample clinical data into tabular format, takes ~15 minutes
import pandas as pd
import xml.etree.ElementTree as et
from settings import params, paths
import os
import numpy as np


xmlDir = "/scratch/pdelangen/projet_these/data_clean/tcga_metadata/"
xmlList = os.listdir(xmlDir)

tabs = []
for i in xmlList:
    if i.endswith(".xml"):
        tab = pd.read_xml(xmlDir + "/" + i)
        # Parsing is not correct and creates two lines, merge on NA values
        tab = tab.iloc[0].where(np.logical_not(pd.isnull(tab.iloc[0])), tab.iloc[1])
        tabs.append(tab)
allTabs = pd.concat(tabs, axis=1).T
allTabs.index = allTabs["bcr_patient_uuid"]
allTabs.to_csv(paths.tempDir + "sample_clinical.tsv", sep="\t", index=None)
# %%
# Query barcode from uuid
manifest = pandas.read_csv("/scratch/pdelangen/projet_these/data_clean/gdc_manifest_clinical.2021-09-27.txt", sep="\t")
fileIDs = list(manifest["id"])
# %%
import pandas as pd
import os 
import sys
sys.path.append("./")
from settings import params, paths
import requests
import json
from io import StringIO


manifest = pd.read_csv(paths.manifest, sep="\t")
filenames = list(manifest["filename"])
fields = pd.read_csv("source/rnaseqAnalysis/downloadCount/queries_samples.txt", header=None)
fields = list(fields.values.ravel())

def query(queries, fieldQuery, fields, endpt):
    filters = {"op":"in",
               "content":{
                    "field": fieldQuery,
                    "value": queries
                }
               }
    params = {"filters": json.dumps(filters),
              "fields": ",".join(fields),
              "format": "TSV",
              "size": str(len(queries)+10)
              }

    response = StringIO(requests.get(endpt, params=params).text)
    df = pd.read_csv(response, sep="\t")
    df.index = df["id"]
    print(len(df), len(queries))
    return df

# Split whole query into smaller chunks or connection timeout
chunkSize = 100
chunks = list(range(0, len(manifest), chunkSize)) + [len(manifest)]
allChunks = []
for i in range(len(chunks)-1):
    queryFiles = filenames[chunks[i]:chunks[i+1]]
    allChunks.append(query(queryFiles, "file_name", fields,"https://api.gdc.cancer.gov/files/"))
allChunks = pd.concat(allChunks)
# %%

# Merge Sample and case dataframe
fileAnnotation = []
for i in range(len(allChunks)):
    matched = allTabs["bcr_patient_uuid"] == allChunks.iloc[i]["cases.0.case_id"]
    if matched.sum() == 1:
        tab = allChunks.iloc[i:i+1].where(np.logical_not(pd.isnull(allChunks.iloc[i:i+1])), allTabs[matched])
        fileAnnotation.append(tab)
    else:
        print("sgdjk bkfg", matched.sum())
fileAnnotation = pd.concat(fileAnnotation, axis=0)
fileAnnotation.to_csv(paths.tempDir + "sample_clinical.tsv", sep="\t", index=None)
fileAnnotation
# %%
