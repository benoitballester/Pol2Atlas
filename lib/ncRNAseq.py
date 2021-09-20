import numpy as np


class atlasRNAseq:
    """
    atlasRNAseq

    

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
    def __init__(self, countDirectory, coordinates):
        allFiles = os.listdir(countDirectory)
        allFiles = [f for f in allFiles if f.endswith("txt.gz")]
        allCounts = []
        keys = []
        scalingFactors = []
        tabsScaling = []
        for f in allFiles:
            key = f[:-7]
            keys.append(key)
            counts = pd.read_csv(countDirectory + f, skiprows=1)
            totalReads = pd.read_csv(f"{parameters.outputDir}/counts/500centroid/{key}.counts.summary",
                                    sep="\t", index_col=0).sum().values[0]
            scalingFactors.append(totalReads)
            tabsScaling.append(pd.read_csv(f"{parameters.outputDir}/counts/500centroid/{key}.counts.summary",
                                    sep="\t", index_col=0))
            allCounts.append(counts.values/totalReads*1e6)
        allCounts = np.concatenate(allCounts, axis=1)
        nzPos = np.sum(allCounts, axis=1) > 0.0016
        self.allCounts = allCounts[nzPos]

    def standardizeCounts(self):
        pass
    
    def jackStrawPCA(self):
        pass
    
    def kaplanMeier(self):
        pass

    def 

        