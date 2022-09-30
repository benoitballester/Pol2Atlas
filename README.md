# Super main title
This is the repository for our article : ""
```
```
## Replicating the results in the paper
All analyses excepted Stratified LD-score regression can be replicated using the main singularity container.
You can also use the provided conda environments, or the singularity recipes which are located in env/. 

If you are interested in a particular figure or data of the paper, you can inspect the metadata file to see which script generated it.

Once all paths have been properly specified in settings/paths.py, all of our analyses can be replicated using :
```
code
```
# Re-using part of our analyses
Here's a quick tutorial on how to re-use some key parts of our code for your analyses.
## Atlas of RNA Polymerase II bound regions
Source code for the Pol II atlas building and analyses are in source/datasetIntegration/ as well as in lib/peakMerge.py .
Our code is easily reusable and applicable to other similar peak-called datasets (CAGE, ChIP-seq, ATAC-seq...) using either command line or more flexibly using Python (lib/peakMerge.py).
Bash command to run consensus peak identification, matrix creation and UMAP/clustering:
```bash
chrSizeFile=[path]
peaksFolder=[path]
fileFormat=[narrowPeak/bed]
outputFolder=[path]
annotationFile=[path]
python peakMerge.py $chrSizeFile \
                    $peaksFolder \
                    $fileFormat \
                    $outputFolder \
                    [--inferCenter] \
                    --annotationFile $annotationFile\
                    --doUmap \
                    --doClustering \
                    --getEnrichedPeaks \
                    --evaluateKNNaccuracy
```
Python script :
```python
from peakMerge import peakMerger
# Parameters
chrSizeFile=[path]
peaksFolder=[path]
fileFormat=[narrowPeak/bed]
outputFolder=[path]
annotationFile=[path]
inferCenter=[False/True]
# Run analyses
merger = peakMerger(chrSizeFile, outputFolder)
merger.mergePeaks(peaksFolder, fileFormat, inferCenter)
merger.umap(transpose=False, annotationFile=annotationFile)
merger.umap(transpose=True, annotationFile=annotationFile)
merger.clusterize(transpose=False, restarts=2, annotationFile=annotationFile)
merger.clusterize(transpose=True, restarts=100, annotationFile=annotationFile)
merger.topPeaksPerAnnot(annotationFile)
merger.knnClassification(annotationFile)
```
For more details see the dedicated github and Read The Doc: peakMerge and rtd.
## GO Enrichment of nearby genes of a subset of genomic regions
Blabla
## Using the Pol II atlas to quantify the non-coding expression in RNA-seq and scRNA-seq
We use featureCounts to count expression in BAM files over Pol II probes. Scripts to download-count-delete files are located in source/rnaseqAnalysis/downloadCount/ . 
Single file counting script is readCountsAtlas.sh. Example usage
```bash

```

If you wish to use the same count transformation pipeline as ours, which yields good results from 100 UMIs single cell to 100M+ reads bulk rna-seq, routines are located in lib/rnaseqFuncs.py and are documented. For R
Users and small counts you can also try the SCTransform library.
Having your raw count matrix stored in a numpy array, you would call:
```python
import rnaseqFuncs
# Filter genes/probes without at least one read in at least 3 samples
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
countsNz = counts[:, nzCounts]
# Compute size factors
sf = rnaseqFuncs.scranNorm(countsNz)
# Compute Pearson residuals and select Highly Variable features
countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf)
residuals = countModel.residuals
hv = countModel.hv
# Compute PCA with optimal number of components
decomp = rnaseqFuncs.permutationPA_PCA(residuals[:, hv], mincomp=2)
```
For 10X single-cell we used cellranger count and mkref (see source/buildRnaSeqFiles/scrnaseq_pipe.sh), and in the case of the Pol II atlas, we normalized by the number of reads and used the same number of PCs as in
the gene-centric dataset.