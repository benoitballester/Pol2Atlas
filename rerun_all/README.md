# Replicating the results of the paper
All analyses can be replicated or rerun with other data using the main singularity container, excepted Stratified LD-score regression which requires a specific singularity container.
You can also use the provided conda environments, or the singularity recipes which are located in env/. 
1/ Replace the paths in settings/paths.py with your own file paths. 
2/ Activate the base singularity environment. (Created with singularity 2.5.2)
3/ Edit the path for the singularity breach
4/ Then launch reproduce.sh.
We only skip the download and read counting over Pol II probes in rna-seq files part as it can take over a month and directly provide the read counting result.

Once all paths have been properly specified in settings/paths.py, all of our analyses can be replicated using :
```
sh rerun_all/reproduce.sh [cores]
```
# Re-using part of our analyses
Here's a quick tutorial on how to re-use some key parts of our code for your analyses. For additional information please open an issue.
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
# Use --inferCenter if you have access to peak summit info
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
chrSizeFile="path"
peaksFolder="path"
fileFormat="narrowPeak/bed"
outputFolder="path"
annotationFile="path"
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
## GSEA of nearby genes of a subset of genomic regions
We use a custom tool to perform GSEA on genomic regions. It can be used with any gene set annotation (given in GMT format) or annotated genome (gtf format). See the docstring for more details.
```python
from pyGREATglm import pyGREAT
gmtFile="path" # Or list of paths (gene sets will be concatenated)
geneFile="path"
chrFile="path"
# One object per gene set
gsea_obj = pyGREAT(gmtFile, geneFile, chrFile)
import pyranges as pr
allRegions = pr.read_bed("path")
subsetRegions = pr.read_bed("path")
results = gsea_obj.findEnriched(subsetRegions, allRegions)
# Alternatively, background regions can be discarded 
# to assume query regions can lie equiprobably on the whole genome
results_nobg = gsea_obj.findEnriched(subsetRegions)
# Results can be plotted with :
gsea_obj.plotEnrichs(results)
gsea_obj.clusterTreemap(results)
```
## Using the Pol II atlas to quantify the non-coding expression in RNA-seq and scRNA-seq
We use featureCounts to count expression in BAM files over Pol II probes. Scripts to download-count-delete files are located in source/rnaseqAnalysis/downloadCount/. 

If you wish to use the same count transformation pipeline as ours, which yields good results from 100 UMIs single cell to 100M+ reads bulk rna-seq, routines are located in lib/rnaseqFuncs.py and are documented. For R Users and small counts you can also try the SCTransform library (which had issues for large counts/library sizes).
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
Alternatively you can also use deviance residuals which appear to work slightly better for small sample sizes or very small counts, but in that case it is recommended to skip hv gene/probe selection and rescale PCA components to unit variance.
```python
import rnaseqFuncs
# Filter genes/probes without at least one read in at least 3 samples
nzCounts = rnaseqFuncs.filterDetectableGenes(counts, readMin=1, expMin=3)
countsNz = counts[:, nzCounts]
# Compute size factors (alternatively just divide by library size)
sf = rnaseqFuncs.scranNorm(countsNz)
# Compute deviance residuals
countModel = rnaseqFuncs.RnaSeqModeler().fit(countsNz, sf, residuals="deviance")
residuals = countModel.residuals
# Compute PCA with optimal number of components
decomp = rnaseqFuncs.permutationPA_PCA(residuals, mincomp=2, whiten=True)
```
For 10X single-cell we used cellranger count and mkref (see source/buildRnaSeqFiles/scrnaseq_pipe.sh). We used the deviance residuals here.