import sys
sys.path.append("./")
import numpy as np
from settings import params, paths
import pyranges as pr
import pandas as pd
from lib.utils import utils
import os

utils.createDir(paths.outputDir + "homer_motifs/")

def find_files_with_suffix(directory, suffix):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(suffix):
                matches.append(os.path.join(root, filename))
    return matches

# Example usage:
directory_path = paths.outputDir  # Replace with your desired directory path
file_suffix = '.bed'  # Replace with your desired file suffix

matching_files = find_files_with_suffix(directory_path, file_suffix)
folder_names = [f[len(paths.outputDir):-4].replace("/","_").replace(" ","_") for f in matching_files]
matching_names = dict(zip(folder_names, matching_files))

print()
rule all:
    input:
        expand(paths.outputDir +  "homer_motifs/{folder}/knownResults.txt", 
                folder=folder_names)

rule homer:
    threads:
        1
    output:
        paths.outputDir +  "homer_motifs/{folder_names}/knownResults.txt"
    params: 
        files = lambda wildcard : matching_names[wildcard[0]]
    shell:
        """
        findMotifsGenome.pl {params.files} {paths.hg38_homer} \
                            {paths.outputDir}/homer_motifs/{wildcards.folder_names}/ \
                            -p 1 -nomotif
        """