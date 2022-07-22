# %%
import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynndescent
import seaborn as sns
import umap
from matplotlib.patches import Patch
from scipy.io import mmwrite
from scipy.signal import argrelextrema, oaconvolve
from scipy.signal.windows import gaussian
from scipy.sparse import csr_matrix
from scipy.stats import binom, hypergeom, mannwhitneyu

try:
    from utils import matrix_utils, overlap_utils, plot_utils
except ModuleNotFoundError:
    from .utils import plot_utils, matrix_utils, overlap_utils

import pickle
import pyranges as pr
import scipy.cluster.hierarchy as hierarchy
from fastcluster import linkage_vector
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.multitest import multipletests

sys.setrecursionlimit(100000)

class peakMerger:
    """
    peakMerger

    Find consensus peaks from multiple peak calling files, and has methods
    to perform data exploration analyses.

    Parameters
    ----------
    genomeFile: string
        Path to tab separated (chromosome, length) annotation file.

    scoreMethod: "binary" or integer (optional, default "binary")
        If set to binary, will use a binary matrix (peak is present or absent).
        Otherwise if set to an integer this will be the column index (0-based)
        used as score. Usually not recommended but the code has fallbacks
        for the non-binary case.

    outputPath: string (optional, default None)
        Folder to export results (matrix, plots...). If not specified will not
        write any results.

    Attributes
    -----------------
    matrix: ndarray of shape (num consensuses, num experiments)
        The experiment-consensus peak presence matrix. It is
        a binary matrix unless the score method has been changed.

    consensuses: pandas DataFrame in bed-like format
        The genomic locations matching the rows (consensuses) of 
        the matrix attribute.
    
    labels: list of strings of size (num experiments)
        The file names matching the columns (experiments) of
        the matrix attribute.

    clustered: (ndarray of shape (num consensuses), ndarray of shape (num experiments))
        Two arrays (first are the consensuses, second are the experiments).
        Index of the cluster each sample belongs to.

    embedding: (ndarray of shape (num consensuses, 2), ndarray of shape (num experiments, 2))
        Two arrays (first are the consensuses, second are the experiments).
        Position of the points in the UMAP 2D space.

    """
    def __init__(self, genomeFile, outputPath=None, scoreMethod="binary"):
        self.score = scoreMethod
        self.outputPath = outputPath
        try:
            os.mkdir(self.outputPath)
        except:
            pass
        if not scoreMethod == "binary":
            self.score = int(scoreMethod)
        with open(genomeFile, 'r') as f:
            self.chrLen = dict()
            self.genomeSize = 0
            for l in f:
                l = l.rstrip("\n").split("\t")
                self.chrLen[l[0]] = int(l[1])
                self.genomeSize += int(l[1])    
        self.embedding = [None, None]
        self.clustered = [None, None]
        self.matrix = None
        self.consensuses = None
        self.labels = None
    
    def save(self):
        with open(self.outputPath + "merger", "wb") as f:
            pickle.dump(self, f, protocol=4)

    def mergePeaks(self, folderPath, fileFormat, inferCenter=False, forceUnstranded=False, 
                  sigma="auto_1", perPeakDensity=False, perPeakMultiplier=0.5,
                  minOverlap=2):
        """
        Read peak called files, generate consensuses and the matrix.

        Parameters
        ----------
        folderPath: string
            Path to either folder with ONLY the peak files or comma separated
            list of files. Accepts compressed files.
        
        fileFormat: "bed" or "narrowPeak"
            Format of the files being read. 
            Bed file format assumes the max signal position to be at the 6th column 
            (0-based) in absolute coordinates.
            The narrowPeak format assumes the max signal position to be the 9th column 
            with this position being relative to the start position.

        inferCenter: boolean (optional, default False)
            If set to true will use the position halfway between start and end 
            positions. Enable this only if the summit position is missing.

        forceUnstranded: Boolean (optional, default False)
            If set to true, assumes all peaks are not strand-specific.

        sigma: float or "auto_1" or "auto_2" (optional, default "auto_1")
            Size of the gaussian filter (lower values = more separation).
            Only effective if perPeakDensity is set to False. 
            "auto_1" automatically selects the filter width at 1/8th of the average peak size.
            "auto_2" automatically selects the filter width based on the peak 
            count and genome size.
        
        perPeakDensity: Boolean (optional, default False)
            If set to false will perform a gaussian filter along the genome (faster),
            assuming all peaks have roughly the same size.
            If set to true will create the density curve per peak based on each peak
            individual size. This is much more slower than the filter method.
            May be useful if peaks are expected to have very different sizes. Can
            also be faster when the number of peaks is small.

        perPeakMultiplier: float (optional, default 0.25)
            Only effective if perPeakDensity is set to True. Adjusts the width of 
            the gaussian fitted to each peak (lower values = more separation).
        
        minOverlap: integer (optional, default 2)
            Minimum number of peaks required at a consensus.
        

        """
        alltabs = []
        if os.path.isdir(folderPath):
            files = os.listdir(folderPath)
        else:
            files = folderPath.split(",")
        for f in files:
            fmt = fileFormat
            if not fmt in ["bed", "narrowPeak"]:
                raise TypeError(f"Unknown file format : {fmt}")
            # Read bed format
            if fmt == "bed":
                if inferCenter:
                    usedCols = [0,1,2,5]
                else:
                    usedCols = [0,1,2,5,6]
                if not self.score == "binary":
                    usedCols.append(self.score)
                tab = pd.read_csv(folderPath + "/" + f, sep="\t", header=None, 
                                  usecols=usedCols)
                tab = tab[usedCols]
                if self.score == "binary":
                    tab[5000] = 1
                tab.columns = np.arange(len(tab.columns))
                tab[0] = tab[0].astype("str", copy=False)
                tab[3].fillna(value=".", inplace=True)
                if inferCenter:
                    tab[5] = tab[4]
                    tab[4] = ((tab[1]+tab[2])*0.5).astype(int)
                tab[6] = f
                if self.score == "binary":
                    tab[5] = [1]*len(tab)
                alltabs.append(tab)
            elif fmt == "narrowPeak":
                if inferCenter:
                    usedCols = [0,1,2,5]
                else:
                    usedCols = [0,1,2,5,9]
                if not self.score == "binary":
                    usedCols.append(self.score)
                tab = pd.read_csv(folderPath + "/" + f, sep="\t", header=None, 
                                  usecols=usedCols)
                tab = tab[usedCols]
                if self.score == "binary":
                    tab[5000] = 1
                tab.columns = np.arange(len(tab.columns))
                tab[0] = tab[0].astype("str", copy=False)
                tab[3].fillna(value=".", inplace=True)
                if inferCenter:
                    tab[5] = tab[4]
                    tab[4] = ((tab[1]+tab[2])*0.5).astype(int, copy=False)
                else:
                    tab[4] = (tab[1] + tab[4]).astype(int, copy=False)
                tab[6] = f
                alltabs.append(tab)
        # Concatenate files
        self.df = pd.concat(alltabs)
        self.numElements = len(self.df)
        self.avgPeakSize = np.mean(self.df[2] - self.df[1])
        # Check strandedness
        if forceUnstranded == True:
            self.df[3] = "."
            self.strandCount = 1
        else:
            # Check if there is only stranded or non-stranded elements
            strandValues = np.unique(self.df[3])
            self.strandCount = len(strandValues)
            if self.strandCount > 2:
                raise ValueError("More than two strand directions !")
            elif self.strandCount == 2 and "." in strandValues:
                raise ValueError("Unstranded and stranded values !")
        # Factorize experiments
        self.df[6], self.labels = pd.factorize(self.df[6])
        # Split per strand
        self.df = dict([(k, x) for k, x in self.df.groupby(3)])
        ########### Peak separation step ########### 
        # Compute sigma if automatic setting
        if sigma == "auto_1":   
            sigma = self.avgPeakSize/8.0
        elif sigma == "auto_2":
            l = self.genomeSize/self.numElements*self.strandCount
            sigma = np.log(10.0) * l / np.sqrt(2.0)
        else:
            sigma = float(sigma)
        if perPeakDensity:
            sigma = perPeakMultiplier
        windowSize = int(8*sigma)+1
        sepPerStrand = {}
        sepIdxPerStrand = {}
        # Iterate for each strand
        for s in self.df.keys():
            # Split peaks per chromosome
            posPerChr = dict([(k, x.values[:, [1,2,4]].astype(int)) for k, x in self.df[s].groupby(0)])
            # Iterate over all chromosomes
            sepPerStrand[s] = {}
            sepIdxPerStrand[s] = {}
            for chrName in posPerChr.keys():
                # Place peak on the genomic array
                try:
                    currentLen = self.chrLen[str(chrName)]
                except KeyError:
                    print(f"Warning: chromosome {str(chrName)} is not in genome annotation and will be removed")
                    continue
                array = np.zeros(currentLen, dtype="float32")
                peakIdx = posPerChr[chrName]
                np.add.at(array, peakIdx[:, 2],1)
                if not perPeakDensity:
                    # Smooth peak density
                    smoothed = oaconvolve(array, gaussian(windowSize, sigma), "same")
                    # Split consensuses
                    separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
                else:
                    smoothed = np.zeros(currentLen, dtype="float32")
                    for i in range(len(peakIdx)):
                        peakSigma = (peakIdx[i, 1] - peakIdx[i, 0])*sigma
                        windowSize = int(8*peakSigma)+1
                        start = max(peakIdx[i, 2] - int(windowSize/2), 0)
                        end = min(peakIdx[i, 2] + int(windowSize/2) + 1, currentLen)
                        diffStart = max(-peakIdx[i, 2] + int(windowSize/2), 0)
                        diffEnd = windowSize + min(currentLen - peakIdx[i, 2] - int(windowSize/2) - 1, 0)
                        smoothed[start:end] += gaussian(windowSize, peakSigma)[diffStart:diffEnd]
                    separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
                separators = separators[np.where(np.ediff1d(separators) != 1)[0]+1]    # Removes consecutive separators (because less-equal comparison)
                separators = np.insert(separators, [0,len(separators)], [0, currentLen])        # Add start and end points
                # Genome position separators
                sepPerStrand[s][chrName] = separators
                # Peak index separator
                array = array.astype("int16", copy=False)
                sepIdxPerStrand[s][chrName] = np.cumsum([np.sum(array[separators[i]: separators[i+1]]) for i in range(len(separators)-1)], dtype="int64")
                del array
        ########### Create matrix and consensus genomic locations ########### 
        self.matrix = []
        self.consensuses = []
        j = 0
        # Iterate over each strand
        for s in self.df.keys():
            self.df[s].sort_values(by=[0, 4], inplace=True)
            posPerChr = dict([(k, x.values) for k, x in self.df[s].groupby(0)])
            # Iterate over each chromosome
            for chrName in posPerChr.keys():
                try:
                    separators = sepPerStrand[s][chrName]
                except:
                    continue
                splits = np.split(posPerChr[chrName], sepIdxPerStrand[s][chrName])
                for i in range(len(splits)):
                    currentConsensus = splits[i]
                    # Exclude consensuses that are too small
                    if len(currentConsensus) < minOverlap:
                        continue
                    currentSep = separators[i:i+2]
                    # Setup consensuses coordinates
                    consensusStart = max(np.min(currentConsensus[:,1]), currentSep[0])
                    consensusEnd = min(np.max(currentConsensus[:,2]), currentSep[1])
                    consensusCenter = int(np.mean(currentConsensus[:,4]))
                    # Mean value of present features
                    meanScore = len(currentConsensus)
                    # Assign scores for each experiment to the current consensus
                    features = np.zeros(len(self.labels), dtype="float32")
                    features[currentConsensus[:, 6].astype(int)] = currentConsensus[:, 5].astype(float)
                    # Add consensus to the score matrix and to the genomic locations
                    self.matrix.append(features)
                    data = [chrName, consensusStart, consensusEnd, j, 
                            meanScore, s, consensusCenter, consensusCenter + 1]
                    self.consensuses.append(data)
                    j += 1
        self.matrix = np.array(self.matrix)
        if self.score == "binary":
            self.matrix = self.matrix.astype(bool)
        self.consensuses = pd.DataFrame(self.consensuses)
        self.save()


    def writePeaks(self):
        """
        Write matrix, datasets names and consensuses genomic locations. 
        The matrix (consensuses, datasets) uses a sparse matrix market format 
        and is saved into "matrix.mtx".
        The dataset names corresponding to the rows are saved in "datasets.txt"
        The genomic locations associated to each consensus are located in "consensuses.bed"

        Parameters
        ----------

        """
        self.consensuses.to_csv(self.outputPath + "consensuses.bed", 
                                sep="\t", header=False, index=False)
        mmwrite(self.outputPath + "matrix.mtx", csr_matrix(self.matrix.astype(float)))
        pd.DataFrame(self.labels).to_csv(self.outputPath + "datasets.txt", 
                                         sep="\t", header=False, index=False)
        with open(self.outputPath + "dataset_stats.txt", "w") as f:
            f.write("Average peak size\t" + str(self.avgPeakSize) + "\n")
            f.write("Number of Peaks\t" + str(self.numElements) + "\n")
            f.write("Number of consensus peaks\t" + str(self.matrix.shape[0]) + "\n")
            f.write("Number of experiments\t" + str(self.matrix.shape[1]) + "\n")
            f.write("Genome size\t" + str(self.genomeSize) + "\n")


    def pseudoHC(self, annotationFile=None, metric="dice", kMetaSamples=50000, method="ward", figureFmt="pdf"):
        orderRows = matrix_utils.threeStagesHC(self.matrix.T, metric)
        orderCols = matrix_utils.threeStagesHC(self.matrix, metric)
        plot_utils.plotHC(self.matrix, self.labels, annotationFile, rowOrder=orderRows, colOrder=orderCols)
        plt.savefig(self.outputPath + f"pseudoHC.{figureFmt}", bbox_inches='tight')
        plt.show()

    def umap(self, transpose, altMatrix=None, annotationFile=None, annotationPalette=None, metric="auto", k=30, figureFmt="pdf", reDo=False):
        """
        Performs UMAP dimensionnality reduction on the matrix, and plot results 
        with the annotation.

        Parameters
        ----------
        transpose: Boolean
            If set to true will perform UMAP on the experiments, otherwise on the 
            consensuses.

        annotationFile: None or string (optionnal, default None)
            Path to a tab separated file, with each line linking a file name 
            (first column)to its annotation (column named 'Annotation', case sensitive)

        altMatrix: None or array-like (optional, default None)
            Alternative representation of the data matrix, which will be used as a 
            for UMAP. For example, PCA or Autoencoder latent space. Array shape
            has to match the merger matrix.

        metric: "auto" or string (optional, default "auto")
            Metric used by umap. If set to "auto", it will use the dice distance
            if the peak scoring is set to binary, or pearson correlation otherwise. 
            See the UMAP documentation for a list of available metrics.

        k : integer (optional, default 15)
            Number of nearest neighbors used by UMAP

        figureFmt: string (optional, default "pdf")
            Format of the output figures.

        reDo: boolean (optional, default "False")
            If it set to false, it will not re do the UMAP transform if it has already
            been done. This is useful if you just want to change the annotation of the samples.
            If set to True it will re-run UMAP with the new parameters.

        Returns
        -------
        embedding: ndarray
            Position of the points in the UMAP 2D space.
        """
        # Autoselect metric
        if metric == "auto":
            if self.score == "binary":
                metric = "dice"
                if transpose:
                    metric = "yule"
            else:
                metric = "correlation"
        # Load annotation file
        annotations = np.zeros(len(self.matrix.T), "int64")
        eq = ["Non annotated"]
        if not annotationFile == None:
            annotationDf = pd.read_csv(annotationFile, sep="\t", index_col=0)
            annotations, eq = pd.factorize(annotationDf.loc[self.labels]["Annotation"],
                                           sort=True)
            if np.max(annotations) >= 18 and annotationPalette is None:
                print("Warning : Over 18 annotations, using random colors instead of a palette")
        # Run UMAP
        if transpose:
            if reDo or (self.embedding[1] is None):
                self.embedderT = umap.UMAP(n_neighbors=k, min_dist=0.5, metric=metric, random_state=42)
                if altMatrix is None:
                    self.embedding[1] = self.embedderT.fit_transform(self.matrix.T)
                else:
                    self.embedding[1] = self.embedderT.fit_transform(altMatrix)
            if annotationPalette is None:
                palette, colors = plot_utils.getPalette(annotations)
            else:
                palette, colors = plot_utils.applyPalette(annotationDf.loc[self.labels]["Annotation"], 
                                                          eq, annotationPalette)
        else:
            if reDo or (self.embedding[0] is None):
                embedder = umap.UMAP(n_neighbors=k, min_dist=0.0, metric=metric,random_state=42)
                if altMatrix is None:
                    self.embedding[0] = embedder.fit_transform(self.matrix)
                else:
                    self.embedding[0] = embedder.fit_transform(altMatrix)
            # Annotate each point according to strongest mean annotation signal in consensus
            signalPerCategory = np.zeros((np.max(annotations)+1, len(self.embedding[0])))
            signalPerAnnot = np.array([np.sum(self.matrix[:, i == annotations]) for i in range(np.max(annotations)+1)])
            for i in range(np.max(annotations)+1):
                signalPerCategory[i, :] = np.sum(self.matrix[:, annotations == i], axis=1) / signalPerAnnot[i]
            signalPerCategory /= np.sum(signalPerCategory, axis=0)
            maxSignal = np.argmax(signalPerCategory, axis=0)
            entropy = np.sum(-signalPerCategory*np.log(signalPerCategory+1e-15), axis=0)
            normEnt = entropy / (-np.log(1.0/signalPerCategory.shape[0]+1e-15))
            gini = (1 - np.sum(np.power(1e-7+signalPerCategory/(1e-7+np.sum(signalPerCategory,axis=0)), 2),axis=0))
            # Retrieve colors based on point annotation
            if annotationPalette is None:
                palette, colors = plot_utils.getPalette(maxSignal)
            else:
                corresp = eq[maxSignal]
                palette, colors = plot_utils.applyPalette(corresp, 
                                                          eq, annotationPalette)
            # Grey out colors of points with varied dataset annotation
            colors = (1.0 - gini[:,None]) * colors + gini[:,None] * 0.5
        # Plot UMAP (scaling point size with dataset size)
        plt.figure(figsize=(10,10), dpi=500)
        if transpose:
            plot_utils.plotUmap(self.embedding[1], colors)
        else:
            plot_utils.plotUmap(self.embedding[0], colors)
        plt.xlabel("UMAP 1", fontsize=8)
        plt.ylabel("UMAP 2", fontsize=8)
        plt.tick_params(
            axis='both', 
            which='both',    
            bottom=False,   
            top=False,         
            left=False,
            labelleft=False,
            labelbottom=False) 
        patches = []
        for i in range(len(eq)):
            legend = Patch(color=palette[i], label=eq[i])
            patches.append(legend)
        plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=6)
        if self.outputPath is not None:
            if transpose:
                pd.DataFrame(self.embedding[int(transpose)]).to_csv(self.outputPath + f"embedding_experiments.txt", header=False, index=False, sep="\t")
                plt.savefig(self.outputPath + f"umap_experiments.{figureFmt}", bbox_inches='tight')
            else:
                pd.DataFrame(self.embedding[int(transpose)]).to_csv(self.outputPath + f"embedding_consensuses.txt", header=False, index=False, sep="\t")
                plt.savefig(self.outputPath + f"umap_consensuses.{figureFmt}", bbox_inches='tight')
        plt.show()
        self.save()
        return self.embedding[int(transpose)]


    def clusterize(self, transpose, altMatrix=None, metric="auto", r=1.0, k="auto", 
                  method="SNN", restarts=1, annotationFile=None, annotationPalette=None, 
                  figureFmt="pdf", reDo=False):
        """
        Performs graph based clustering on the matrix.

        Parameters
        ----------
        transpose: Boolean
            If set to true will perform clustering on the experiments, otherwise on the consensuses.

        metric: "auto" or string (optional, default "auto")
            Metric used by umap. If set to "auto", it will use the dice ditance
            if the peak scoring is set to binary, or pearson correlation otherwise. 
            See the pynnDescent documentation for a list of available metrics.

        altMatrix: None or array-like (optional, default None)
            Alternative representation of the data matrix, which will be used as a 
            for clustering. For example, PCA or Autoencoder latent space. Array shape
            has to match the merger matrix.

        r: float (optional, default 0.4)
            Resolution parameter of the graph partitionning algorithm. Lower values = less clusters.

        k: "auto" or integer (optional, default "auto")
            Number of nearest neighbors used to build the NN graph.
            If set to auto uses 2*numPoints^0.25 neighbors as a rule of thumb, as too few 
            NN with a lot of points can create disconnections in the graph.

        method: {"SNN", "KNN"} (optional, default "SNN")
            If set to "SNN", it will perform the Shared Nearest Neighbor Graph 
            clustering variant, where the edges of the graph are weighted according 
            to the number of shared nearest neighbors between two nodes. If set to "KNN",
            all edges are equally weighted. SNN can produce a more refined clustering 
            but it can also hallucinate some clusters.

        restarts: integer (optional, default 1)
            The number of times to restart the graph partitionning algorithm, before keeping 
            the best partition according to the quality function.
        
        annotationFile: None or string (optional, default None)
            Path to a tab separated file, with each line linking a file name 
            (first column) to its annotation (column named 'Annotation', case sensitive)

        figureFmt: string (optional, default "pdf")
            Format of the output figures.

        reDo: boolean (optional, default "False")
            If it set to false, it will not re do the clustering if it has already
            been done. This is useful if you just want to change the annotation of the samples.
            If set to True it will re-run clustering with the new parameters.

        Returns
        -------
        labels: ndarray
            Index of the cluster each sample belongs to.
        """
        if not method in {"SNN", "KNN"}:
            raise TypeError(f"Invalid clustering method : {method}, please use 'SNN' or 'KNN'")
        if metric == "auto":
            if self.score == "binary":
                metric = "dice"
                if transpose:
                    metric = "yule"
            else:
                metric = "correlation"
        if reDo or (self.clustered[int(transpose)] is None):
            if transpose:
                if altMatrix is None:
                    self.clustered[1] = matrix_utils.graphClustering(self.matrix.T, 
                                                                    metric, k, r, method=="SNN",
                                                                    restarts=restarts)
                else:
                    self.clustered[1] = matrix_utils.graphClustering(altMatrix, 
                                                                    metric, k, r, method=="SNN",
                                                                    restarts=restarts)
            else:
                if altMatrix is None:
                    self.clustered[0] = matrix_utils.graphClustering(self.matrix, 
                                                                    metric, k, r, method=="SNN",
                                                                    restarts=restarts)
                else:
                    self.clustered[0] = matrix_utils.graphClustering(altMatrix, 
                                                                    metric, k, r, method=="SNN",
                                                                    restarts=restarts)
        # Get experiment annotation
        if not annotationFile == None:
            annotationDf = pd.read_csv(annotationFile, sep="\t", index_col=0)
            annotations, eq = pd.factorize(annotationDf.loc[self.labels]["Annotation"], sort=True)
            if annotationPalette is None:
                palette, colors = plot_utils.getPalette(annotations)
            else:
                palette, colors = plot_utils.applyPalette(annotationDf.loc[self.labels]["Annotation"], 
                                                          eq, annotationPalette)
            if np.max(annotations) >= 20 and annotationPalette is None:
                print("Warning : Over 18 annotations, using random colors instead of a palette")
            if transpose:
                print("Clustering metrics of the experiments wrt the annotation file :")
                print("ARI : ", adjusted_rand_score(annotations, self.clustered[int(transpose)]))
                print("AMI : ", adjusted_mutual_info_score(annotations, self.clustered[int(transpose)]))
                with open(self.outputPath + "ARI_AMI.txt", "w") as f:
                    f.write("ARI\t" + str(adjusted_rand_score(annotations, self.clustered[int(transpose)])) + "\n" )
                    f.write("AMI\t" + str(adjusted_mutual_info_score(annotations, self.clustered[int(transpose)])) + "\n" )
            else:
                # Evaluate per cluster annotation proportion and plot it for each cluster
                try:
                    os.mkdir(self.outputPath + "/clusters_consensuses_summaries/")
                except:
                    pass
                allProportions = []
                for c in range(self.clustered[int(transpose)].max()+1):
                    inClust = self.clustered[int(transpose)] == c
                    signalPerCategory = np.zeros(np.max(annotations)+1)
                    signalPerAnnot = np.array([np.sum(self.matrix[:, i == annotations]) for i in range(np.max(annotations)+1)])
                    for i in range(np.max(annotations)+1):
                        signalPerCategory[i] = np.sum(self.matrix[inClust][:, annotations == i])/signalPerAnnot[i]
                    maxSignal = np.argmax(signalPerCategory)
                    normSignal = signalPerCategory/signalPerCategory.sum()
                    allProportions.append(normSignal)
                    plt.figure(dpi=500)
                    runningSum = 0
                    for j, p in enumerate(normSignal):
                        plt.barh(0, p, left=runningSum, color=palette[j])
                        runningSum += p
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelleft=False,
                        labelbottom=False) # labels along the bottom edge are off
                    plt.tick_params(axis="x", labelsize=6)
                    plt.tick_params(length=3., width=1.2)
                    plt.xlabel("Fraction of peaks in cluster", fontsize=10)
                    plt.gca().set_aspect(0.1)
                    foldChange = signalPerCategory[maxSignal] / np.mean(signalPerCategory[list(range(len(signalPerCategory))).remove(maxSignal)])
                    numPeaks = np.sum(inClust)
                    plt.title(f'Cluster #{c} ({numPeaks} peaks)\nTop system : {eq[maxSignal]} (Fold change : {int(foldChange*100)/100})', fontsize=12, color=palette[maxSignal], fontweight='bold')
                    plt.savefig(self.outputPath + f"/clusters_consensuses_summaries/{c}.{figureFmt}", bbox_inches='tight')
                    plt.show()
                    plt.close()
                # All clusters summary barplot
                plt.figure(dpi=500)
                # Order clusters by similarity using ward hierarchical clustering
                # Ordered by largest
                link = linkage_vector(allProportions, method="ward")
                rowOrder = hierarchy.leaves_list(link)
                np.savetxt(self.outputPath  + "clusterBarplotOrder.txt", rowOrder)
                for i, o in enumerate(rowOrder):
                    pcts = allProportions[o]
                    runningSum = 0
                    reOrdered = np.argsort(pcts)[::-1]
                    orderedPalette = palette[reOrdered]
                    for j, p in enumerate(pcts[reOrdered]):
                        plt.barh(i, p, left=runningSum, color=orderedPalette[j])
                        runningSum += p
                plt.ylabel(f"{self.clustered[int(transpose)].max()+1} Clusters", fontsize=8)
                plt.xlabel("Origin of peaks (Fraction of cluster)", fontsize=8)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                patches = []
                for i in range(0, min(len(palette), len(eq))):
                    legend = Patch(color=palette[i], label=eq[i])
                    patches.append(legend)
                plt.gca().tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelleft=False,
                        labelbottom=False) # labels along the bottom edge are off
                plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=4)
                plt.savefig(self.outputPath + f"consensuses_summary_ordered.{figureFmt}", bbox_inches='tight')
                plt.show()
                plt.close()
                # Same order
                plt.figure(dpi=500)
                link = linkage_vector(allProportions, method="ward")
                rowOrder = hierarchy.leaves_list(link)
                np.savetxt(self.outputPath  + "clusterBarplotOrder.txt", rowOrder)
                for i, o in enumerate(rowOrder):
                    pcts = allProportions[o]
                    runningSum = 0
                    orderedPalette = palette
                    for j, p in enumerate(pcts):
                        plt.barh(i, p, left=runningSum, color=orderedPalette[j])
                        runningSum += p
                plt.ylabel(f"{self.clustered[int(transpose)].max()+1} Clusters", fontsize=8)
                plt.xlabel("Origin of peaks (Fraction of cluster)", fontsize=8)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                patches = []
                for i in range(0, min(len(palette), len(eq))):
                    legend = Patch(color=palette[i], label=eq[i])
                    patches.append(legend)
                plt.gca().tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelleft=False,
                        labelbottom=False) # labels along the bottom edge are off
                plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                    loc="lower left", mode="expand", ncol=4)
                plt.savefig(self.outputPath + f"consensuses_summary.{figureFmt}", bbox_inches='tight')
                plt.show()
                plt.close()
        palette, colors = plot_utils.getPalette(self.clustered[int(transpose)])
        # Plot clustering on UMAP graph
        plt.figure(figsize=(10,10), dpi=500)
        if transpose:
            if self.embedding[1] is not None:
                plot_utils.plotUmap(self.embedding[1], colors)
        else:
            if self.embedding[0] is not None:
                # a = 0
                plot_utils.plotUmap(self.embedding[0], colors)
        plt.xlabel("UMAP 1", fontsize=8)
        plt.ylabel("UMAP 2", fontsize=8)
        plt.tick_params(
            axis='both', 
            which='both',    
            bottom=False,   
            top=False,         
            left=False,
            labelleft=False,
            labelbottom=False)
        # Save results
        if self.outputPath is not None: 
            if transpose:
                plt.savefig(self.outputPath + f"/umapExperiments_Clusters.{figureFmt}", bbox_inches='tight')
                pd.DataFrame(self.clustered[int(transpose)]).to_csv(self.outputPath + "/clusterExperiments_Labels.txt", sep="\t", header=False, index=False) 
            else:
                plt.savefig(self.outputPath + f"/umapConsensuses_Clusters.{figureFmt}", bbox_inches='tight')
                pd.DataFrame(self.clustered[int(transpose)]).to_csv(self.outputPath + "/clusterConsensuses_Labels.txt", sep="\t", header=False, index=False) 
                # Save a bed file corresponding to all the consensus peaks that belongs to one cluster
                try:
                    os.mkdir(self.outputPath + "/clusters_bed/")
                except:
                    pass
                for i in range(self.clustered[int(transpose)].max()+1):
                    self.consensuses.iloc[self.clustered[int(transpose)] == i].to_csv(f"{self.outputPath}/clusters_bed/cluster_{i}.bed", 
                                                                        sep="\t", header=False, index=False)
        plt.show()
        self.save()
        return self.clustered[int(transpose)]


    def computeOverlapEnrichment(self, catalogFile, labels, name=None):
        """
        Computes an hypergeometric enrichment test on the number of intersections
        for different classes of genomic elements in the catalog (name column)
        and for each class (labels) of consensus peaks.

        Parameters
        ----------
        catalogFile: string
            Elements to find enrichment on.
            PyRanges having the category of the genomic element in the "name" column.
        labels: array like
            The group each consensus peak belongs to. Can be integer or string.
            It will iterate over each unique label. Ex: clustering labels
        name: string
            Name of the output folder (which will already be located in the output directory).
            Will contain pvalues, fold changes and q values tsv files.
        
        Returns
        -------
        pvalues: pandas DataFrame
            pvalues for each label and class of genomic element
        fc: pandas DataFrame
            Fold change for each label and class of genomic element
        qvalues: pandas DataFrame
            Benjamini-Hochberg qvalues for each label and class of genomic element
            
        """
        catalog = pr.read_bed(catalogFile)
        prConsensuses = overlap_utils.dfToPrWorkaround(merger.consensuses)
        enrichments = overlap_utils.computeEnrichForLabels(catalog, prConsensuses, labels)
        names = ["pvalues", "fold_change", "qvalues"]
        try:
            os.mkdir(self.outputPath + "enrichments/")
        except:
            pass
        try:
            os.mkdir(self.outputPath + "enrichments/" + name)
        except:
            print("Warning : Can't create folder with path :'", self.outputPath + "enrichments/" + name, "', folder probably already exists")
        for i, df in enumerate(enrichments):
            df.to_csv(self.outputPath + "enrichments/" + name + f"/{names[i]}.tsv", sep="\t")
        return enrichments


    def topPeaksPerAnnot(self, annotationFile, multitesting="fdr_bh", alpha=0.05):
        """
        For each annotation, find the consensuses with enriched presence in experiments
        belonging to this annotation. Performs a hypergeometric test in the 
        binary case, or a mann-whitney U test otherwise. The statistically 
        significant consensuses for each annotation can be saved in a folder.

        Parameters
        ----------
        annotationFile: string
            Path to a tab separated file, with each line linking a file name 
            (first column) to its annotation (column named 'Annotation', case sensitive)

        multitesting: string or callable (optional, default "fdr_by")
            The multitesting correction method to use. See the 
            statsmodels.stats.multitest documentation for a list of
            available methods. If set to a callable, requires a function
            that takes the p-values as input, then returns a boolean
            array of statistically signifant p-values and the corrected
            pvalues.

        alpha: float
            The FWER or the FDR (depending on the method).

        Returns
        -------
        perAnnotConsensuses: dictionnary of pandas DataFrames
            Each key correspond to an annotation and its associated consensuses.

        perAnnotQvals: dictionnary of ndarrays
            Each key correspond to an annotation and its associated consensuses pvalue.
        """
        annotationDf = pd.read_csv(annotationFile, sep="\t", index_col=0)
        annotations, eq = pd.factorize(annotationDf.loc[self.labels]["Annotation"], sort=True)
        M = self.matrix.sum()
        perAnnotConsensuses = dict()
        perAnnotQvals = dict()
        # For each annotation
        for i in range(len(eq)):
            hasAnnot = annotations == i
            N = self.matrix[:, hasAnnot].sum()
            n = np.sum(self.matrix, axis=1)
            k = np.sum(self.matrix[:, hasAnnot], axis=1)
            if self.score == "binary":
                pvals = hypergeom(M, n, N).sf(k-1)
            else:
                pvals = mannwhitneyu(self.matrix[:, hasAnnot], 
                                          self.matrix[:, np.logical_not(hasAnnot)], 
                                          axis=1, alternative="greater")[1]
            # Multitesting correction
            if type(multitesting) is str:
                qvals = multipletests(pvals, method=multitesting, alpha=alpha)[1]
            else:
                qvals = multitesting(pvals)[1]
            sig = qvals < alpha
            sigConsensuses = self.consensuses.iloc[sig]
            if self.outputPath is not None:
                try:
                    os.mkdir(self.outputPath + "enrichedPerAnnot/")
                except FileExistsError:
                    pass
                sigConsensuses.to_csv(self.outputPath + f"enrichedPerAnnot/{eq[i]}.bed", header=False, index=False, sep="\t")
                pd.DataFrame(qvals[sig]).to_csv(self.outputPath + f"enrichedPerAnnot/{eq[i]}_qvals.bed", header=False, index=False, sep="\t")
            perAnnotQvals[eq[i]] = qvals[sig]
            perAnnotConsensuses[eq[i]] = sigConsensuses
        return perAnnotConsensuses, perAnnotQvals


    def knnClassification(self, annotationFile, metric="auto", k=1, evalMetric="balanced_accuracy"):
        """
        Evaluate classification performance using a k-NN classifier
        with leave-one-out cross-validation.

        Parameters
        ----------
        annotationFile: string
            Path to a tab separated file, with each line linking a file name 
            (first column) to its annotation (column named 'Annotation', case sensitive)

        metric: "auto" or string (optional, default "auto")
            Metric used to build the NN graph. If set to "auto", it will use the Dice distance 
            if the peak scoring is set to binary, or pearson correlation otherwise. 
            See the UMAP documentation for a list of available metrics.

        k: integer (optional, default 1)
            Number of nearest neighbors (excluding itself) used for classification.

        evalMetric: "balanced_accuracy" or callable (optional, default "balanced_accuracy")
            By default uses balanced accuracy, but also accepts a callable which
            takes as arguments (y_true, y_pred).

        Returns
        -------
        metric: float
            The value of the evalMetric.
        knnClassifier: KNeighborsClassifier
            A fitted sklearn KNeighborsClassifier.
        """
        if metric == "auto":
            if self.score == "binary":
                metric = "yule"
            else:
                metric = "correlation"
        index = pynndescent.NNDescent(self.matrix.T, n_neighbors=min(30+k, len(self.matrix.T)-1), 
                                    metric=metric, diversify_prob=0.0, pruning_degree_multiplier=9.0)
        # Exclude itself and select NNs (equivalent to leave-one-out cross-validation)
        # Pynndescent is ran with a few extra neighbors for a better accuracy on ANNs
        nnGraph = index.neighbor_graph[0][:, 1:k+1]
        annotationDf = pd.read_csv(annotationFile, sep="\t", index_col=0)
        annotations, eq = pd.factorize(annotationDf.loc[self.labels]["Annotation"], sort=True)
        pred = []
        for nns in annotations[nnGraph]:
            # Find the most represented annotation in the k nearest neighbors
            pred.append(np.argmax(np.bincount(nns)))
        if evalMetric == "balanced_accuracy":
            score = balanced_accuracy_score(annotations, pred)
        else:
            score = evalMetric(annotations, pred)
        with open(self.outputPath + "knnBalancedAcc.txt", "w") as f:
            f.write(str(score))
        return balanced_accuracy_score(annotations, pred), KNeighborsClassifier(1, metric=metric).fit(self.matrix.T, annotations)



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A simple tool to integrate and analyse peak called experiments. Use the python API for more options and a more flexible usage.")
    parser.add_argument("genomeFile", help="Path to tab separated (chromosome, length) annotation file.", type=str)
    parser.add_argument("folderPath", help="Path to either folder with ONLY the peak files or comma separated list of files. Accepts compressed files.", type=str)
    parser.add_argument("fileFormat", help=
                        """
                        "bed" or "narrowPeak"
                        Format of the files being read.
                        Bed file format assumes the max signal column to be at the 6th column 
                        (0-based) in absolute coordinates.
                        The narrowPeak format assumes this to be the 9th column with this number 
                        being relative to the start position.""", type=str)
    parser.add_argument("outputFolder", help="""
                        Path of a folder (preferably empty) to write into. Will attempt to create the folder
                        if it does not exists.
                        """, type=str)
    parser.add_argument("--annotationFile", help="""
                        Path to a tab separated file, with each line linking a file name 
                        (first column) to its annotation (column named 'Annotation', case sensitive).
                        If it is not supplied UMAP and the clustering will work, but it will
                        not show the dataset annotations.
                        """, type=str, default=None)
    parser.add_argument("--annotationPalette", help="""
                        Path to a tab separated file, with each line linking an annotation
                        (first column) to its color (columns named 'r','g' and 'b', case sensitive).
                        If it is not supplied PeakMerge will find an adequate color palette 
                        accordint to the annotation size.
                        """, type=str, default=None)
    parser.add_argument("--doUmap", help=
                        """
                        If enabled, performs UMAP dimensionality reduction. Use the python
                        API for additional parameters. 
                        """, action="store_true")
    parser.add_argument("--doClustering", help=
                        """
                        If enabled, performs graph clustering. Use the python
                        API for additional parameters. 
                        """, action="store_true")
    parser.add_argument("--getEnrichedPeaks", help=
                        """
                        If enabled, find the peaks most associated with an annotation.
                        Requires an annotation file to be given.
                        """, action="store_true")
    parser.add_argument("--evaluateKNNaccuracy", help=
                        """
                        If enabled, evaluates a 1-NN classifier on the dataset annotated
                        with the annotation file.
                        Requires an annotation file to be given.
                        """, action="store_true")
    parser.add_argument("--forceUnstranded", help=
                        """
                        If enabled, assumes all peaks are not strand-specific.
                        """, action="store_true")
    parser.add_argument("--inferCenter", action="store_true",
                        help="If enabled will use the position halfway between start and end positions as summit. Enable this only if the summit position is missing.")
    parser.add_argument("--sigma", help=
                        """
                        float or "auto_1" or "auto_2" (optional, default "auto_1")
                        Size of the gaussian filter for the peak separation 
                        (lower values = more separation).
                        Only effective if perPeakDensity is set to False. 
                        "auto_1" automatically selects the filter width based on the average
                        peak size.
                        "auto_2" automatically selects the filter width based on the peak 
                        count and genome size.
                        """, default="auto_1")
    parser.add_argument("--scoreMethod", 
                        help="""
                        "binary" or integer (default "binary")
                        If set to binary, will use a binary matrix (peak is present or absent).
                        Otherwise if set to an integer this will be the column index (0-based)
                        used as score.""", default="binary")
    parser.add_argument("--minOverlap", help=
                        """
                        integer (optional, default 2)
                        Minimum number of peaks required at a consensus.
                        2 is the default but 1 can be used on smaller datasets.
                        """, type=int, default=2)
    parser.add_argument("--figureFormat", help=
                        """
                        Format of the output figures (default pdf).
                        """, default="pdf")
    args = parser.parse_args()
    try:
        os.mkdir(args.outputFolder)
    except:
        pass
    with open(args.outputFolder + "command.sh", "w") as f:
        f.write(" ".join(sys.argv))
    # Run analyses
    merger = peakMerger(args.genomeFile, outputPath=args.outputFolder, 
                        scoreMethod=args.scoreMethod)
    merger.mergePeaks(args.folderPath, forceUnstranded=args.forceUnstranded, 
                      sigma=args.sigma, inferCenter=args.inferCenter, 
                      fileFormat=args.fileFormat, minOverlap=args.minOverlap)
    merger.writePeaks()
    print(f"Got a matrix of {merger.matrix.shape[0]} consensuses and {merger.matrix.shape[1]} experiments")
    if args.doUmap:
        merger.umap(transpose=False, annotationFile=args.annotationFile, 
                    figureFmt=args.figureFormat, annotationPalette=args.annotationPalette)
        merger.umap(transpose=True, annotationFile=args.annotationFile, 
                    figureFmt=args.figureFormat, annotationPalette=args.annotationPalette)
    if args.doClustering:
        merger.clusterize(transpose=False, figureFmt=args.figureFormat,
                          restarts=2, annotationFile=args.annotationFile, annotationPalette=args.annotationPalette)
        merger.clusterize(transpose=True, figureFmt=args.figureFormat,
                          restarts=100, annotationFile=args.annotationFile, annotationPalette=args.annotationPalette)
    if args.getEnrichedPeaks:
        if args.annotationFile is not None:
            merger.topPeaksPerAnnot(args.annotationFile)
        else:
            print("No annotation file given, will skip enriched peak analysis.")
    if args.evaluateKNNaccuracy:
        if args.annotationFile is not None:
            score, knnClassifiery = merger.knnClassification(args.annotationFile)
            print(f"1-NN Balanced accuracy: {score}")
        else:
            print("No annotation file given, will skip 1-NN accuracy.")
    sys.exit()

# %%
