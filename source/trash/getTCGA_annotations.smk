# %%
import pandas
import os 
import sys
sys.path.append("./")
from settings import params, paths


manifest = pandas.read_csv("/scratch/pdelangen/projet_these/data_clean/gdc_manifest_clinical.2021-09-27.txt", sep="\t")

fileIDs = list(manifest["id"])
names = list(manifest["filename"])


dlDir = "/scratch/pdelangen/projet_these/data_clean/tcga_metadata"
idNameMap = dict(zip(names, fileIDs))

rule All:
    input:
        expand(dlDir + "/{final}", final=names)
    
rule dlCount:
    threads:
        1
    params:
        fileIDs = lambda wildcard : idNameMap[wildcard[0]]
    output:
        dlDir + "/{names}"
    shell:
        """
        # Download bam file
        {paths.gdcClientPath} download {params.fileIDs} \
                                 -t {paths.tokenPath} \
                                 -d {dlDir}
        mv {dlDir}/{params.fileIDs}/{wildcards.names} {dlDir}/{wildcards.names}
        rm -rf {dlDir}/{params.fileIDs}
        """