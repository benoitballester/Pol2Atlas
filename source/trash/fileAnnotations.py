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
