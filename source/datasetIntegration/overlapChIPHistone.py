# %%
import pandas as pd
import numpy as np
import deeptools.heatmapper 
from settings import params, paths

histonePathHeart27Ac = "/scratch/pdelangen/projet_these/data_clean/H3K27Ac_Chipseq/h3k27ac_heart_left_ventricule.bigWig"
histonePathLiver27Ac = "/scratch/pdelangen/projet_these/data_clean/H3K27Ac_Chipseq//h3k27ac_liver.bigWig"
histonePathTcell27Ac = "/scratch/pdelangen/projet_these/data_clean/H3K27Ac_Chipseq//h3k27ac_T_cells.bigWig"
histonePathHeart27Me3 = "/scratch/pdelangen/projet_these/data_clean/Dnase/ENCFF169GLM_heart.bigWig"
histonePathLiver27Me3 = "/scratch/pdelangen/projet_these/data_clean/Dnase/ENCFF556YQA_liver.bigWig"
histonePathTcell27Me3 = "/scratch/pdelangen/projet_these/data_clean/Dnase/ENCFF677BPI_Tcells.bigWig"

allPeaks = pd.read_csv(paths.outputDir + "consensuses.bed", sep="\t", header=None)
allPeaks = allPeaks[[0,6,7]] # Summits only
clusts = np.loadtxt(paths.outputDir + "clusterConsensuses_Labels.txt").astype(int)

# %%
def subSampleFromClust(peaks, clusts=None, query=None, invert=False, s=10000, suffix=""):
    if query == None:
        clust = peaks
    else: 
        inClust = query == clusts
        if invert:
            inClust = np.logical_not(inClust)
        clust = allPeaks[inClust]
    subSample = np.random.choice(len(clust), min(s, len(clust)))
    outputPath = paths.tempDir + f"subSample_{str(query)}_{suffix}.bed"
    if invert:
        outputPath = paths.tempDir + f"subSample_not{str(query)}_{suffix}.bed"
    clust.iloc[subSample].to_csv(outputPath, 
                                 sep="\t", header=None, index=None)
    return outputPath

beds = []
beds.append(subSampleFromClust(allPeaks, clusts=clusts, query=9))
beds.append(subSampleFromClust(allPeaks, clusts=clusts, query=15))
beds.append(subSampleFromClust(allPeaks, clusts=clusts, query=4))
# %%
import subprocess
cmd = f"""computeMatrix reference-point \
        -S {histonePathHeart27Ac} {histonePathLiver27Ac} {histonePathTcell27Ac} {histonePathHeart27Me3} {histonePathLiver27Me3} {histonePathTcell27Me3} \
        -R {" ".join(beds)} \
        -b 5000 -a 5000 -p 50 \
        -o {paths.tempDir}cardiovascularVsNotCard.mat"""
subprocess.run(cmd.split())
# %%
cmdPlot = f"""plotProfile -m {paths.tempDir}cardiovascularVsNotCard.mat \
             --averageType mean \
             --yAxisLabel \
             --samplesLabel \
             -z \
             --numPlotsPerRow 3 \
             --refPointLabel \
             -o {paths.outputDir}/epigenetic/clusters_vs_histoneATAC.pdf"""
cmdSplit = cmdPlot.split()
cmdSplit.insert(6, "Mean fold change over input")
cmdSplit.insert(8, "H3K27Ac Heart")
cmdSplit.insert(9, "H3K27Ac Liver")
cmdSplit.insert(10, "H3K27Ac T Cells")
cmdSplit.insert(11, "ATAC-seq Heart")
cmdSplit.insert(12, "ATAC-seq Liver")
cmdSplit.insert(13, "ATAC-seq T Cells")
cmdSplit.insert(15, "'Cardiovascular' cluster")
cmdSplit.insert(16, "'Hepatic' cluster")
cmdSplit.insert(17, "'Lymphoid' cluster")
cmdSplit.insert(21, "Pol II consensus centroid")
subprocess.run(cmdSplit)

# %%
'''
import subprocess
cmd = f"""computeMatrix reference-point \
        -S {histonePathTcell27Me3} \
        -R {" ".join(beds)} \
        -b 5000 -a 5000 -p 50 \
        -o {paths.tempDir}dnase.mat"""
subprocess.run(cmd.split())

cmdPlot = f"""plotProfile -m {paths.tempDir}dnase.mat \
             --averageType median \
             --yAxisLabel \
             --samplesLabel \
             -z \
             --numPlotsPerRow 3 \
             --refPointLabel \
             -o {paths.tempDir}cardHistvsOthers_median.pdf"""
cmdSplit = cmdPlot.split()
cmdSplit.insert(6, "Median fold change over input")
cmdSplit.insert(8, "ATAC-seq Heart")
cmdSplit.insert(10, "'Cardiovascular' cluster")
cmdSplit.insert(11, "'Hepatic' cluster")
cmdSplit.insert(12, "'Lymphoid' cluster")
cmdSplit.insert(16, "Pol II consensus centroid")
subprocess.run(cmdSplit)
'''
# %%
import subprocess
shuffledReg = pd.read_csv(paths.tempDir + "background.bed", sep="\t", header=None)
meanPos = (shuffledReg[2]*0.5 + shuffledReg[1]*0.5).astype(int)
shuffledReg[[1,2]] = np.array([meanPos, meanPos+1]).T
# %%
consFile = "/scratch/pdelangen/projet_these/data_clean/cons/hg38.phyloP100way.bw"
cmd = f"""computeMatrix reference-point \
        -S {consFile} \
        -R {subSampleFromClust(allPeaks,s=50000)} \
        -b 5000 -a 5000 -p max \
        --missingDataAsZero \
        -o {paths.tempDir}polII.mat"""
subprocess.run(cmd.split())
cmdPlot = f"""plotHeatmap -m {paths.tempDir}polII.mat \
             --averageType mean \
             --perGroup \
             -o {paths.outputDir}/epigenetic/consNoClust.pdf"""
subprocess.run(cmdPlot.split())
# %%
consFile = "/scratch/pdelangen/projet_these/data_clean/cons/hg38.phyloP100way.bw"
cmd = f"""computeMatrix reference-point \
        -S {consFile} \
        -R {subSampleFromClust(shuffledReg,s=50000, suffix="Random")} \
        -b 5000 -a 5000 -p max \
        --missingDataAsZero \
        -o {paths.tempDir}random.mat"""
subprocess.run(cmd.split())
cmdPlot = f"""plotHeatmap -m {paths.tempDir}random.mat \
             --averageType mean \
             --perGroup \
             -o {paths.outputDir}/epigenetic/consRandom.pdf"""
subprocess.run(cmdPlot.split())
# %%
