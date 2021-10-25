import pandas as pd
import numpy as np
import pyranges as pr
from statsmodels.stats.multitest import fdrcorrection
from scipy.sparse import coo_matrix, csr_matrix
from .utils import overlap_utils

class pyGREAT:
    """
    doc
    """
    def __init__(self, oboFile, geneFile, geneGoFile, gtfGeneCol="gene_name"):
        self.gtfGeneCol = "gene_name"
        # Parse GO terms
        allLines = []
        termId = None
        namespace = None
        name = None
        with open(oboFile) as f:
            for l in f.readlines():
                if l.startswith("[Term]"):
                    if (not termId == None) and (not namespace == None) and (not name == None):
                        allLines.append((termId, namespace, name))
                    termId = None
                    namespace = None
                    name = None
                elif l.startswith("id"):
                    termId = l.rstrip("\n").split(": ")[1]
                elif l.startswith("namespace"):
                    namespace = l.rstrip("\n").split(": ")[1]
                elif l.startswith("name"):
                    name = l.rstrip("\n").split(": ")[1]
        self.df = pd.DataFrame(allLines)
        self.df.columns = ["id", "namespace", "name"]
        # self.df.set_index("id", inplace=True)
        # Read organism GO gene annotations
        goAnnotation = pd.read_csv(geneGoFile, sep="\t", skiprows=41, header=None)
        # Remove NOT associations
        goAnnotation = goAnnotation[np.logical_not(goAnnotation[3].str.startswith("NOT"))]
        goAnnotation = goAnnotation[[2, 4]]
        goAnnotation.dropna(inplace=True)
        goFull = goAnnotation.merge(self.df, left_on=4, right_on="id")
        goFull.drop(4, 1, inplace=True)
        goFull.rename({2:self.gtfGeneCol}, axis=1, inplace=True)
        # Subset by GO class
        gb = goFull.groupby("namespace")
        self.goClasses = dict([(x,gb.get_group(x)) for x in gb.groups])

        # Read gtf file
        gencode = pr.read_gtf(geneFile)
        gencode = gencode.as_df()
        transcripts = gencode[gencode["Feature"] == "gene"].copy()
        del gencode
        transcripts = transcripts[["Chromosome", "Start", "End", gtfGeneCol, "Strand"]]
        # Reverse positions on opposite strand for convenience
        geneInList = np.isin(transcripts[self.gtfGeneCol], np.unique(goAnnotation[2]), assume_unique=True)
        reversedTx = transcripts.copy()[["Chromosome", "Start", "End", self.gtfGeneCol]][geneInList]
        reversedTx["Start"] = transcripts["Start"].where(transcripts["Strand"] == "+", transcripts["End"])
        reversedTx["End"] = transcripts["End"].where(transcripts["Strand"] == "+", transcripts["Start"])
        # Infered regulatory domain logic
        upstream = 5000
        downstream = 1000
        distal = 1000000
        reversedTx.sort_values(by=["Chromosome", "Start"], inplace=True)
        regPm = reversedTx["Start"] - upstream * np.sign(reversedTx["End"]-reversedTx["Start"])
        regPp = reversedTx["Start"] + downstream * np.sign(reversedTx["End"]-reversedTx["Start"])
        gb = reversedTx.groupby("Chromosome")
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = reversedTx["Chromosome"] == c
            previousReg = np.roll(regPp[inIdx], 1)
            previousReg[0] = 0
            previousReg[-1] = int(1e10)
            nextReg = np.roll(regPm[inIdx], -1)
            nextReg[-1] = int(1e10)
            extendedM = np.maximum(reversedTx["Start"][inIdx] - distal, np.minimum(previousReg, regPm[inIdx]))
            extendedP = np.minimum(reversedTx["Start"][inIdx] + distal, np.maximum(nextReg, regPp[inIdx]))
            reversedTx.loc[reversedTx["Chromosome"] == c, "Start"] = extendedM
            reversedTx.loc[reversedTx["Chromosome"] == c, "End"] = extendedP
        self.geneRegulatory = reversedTx
        # Merge regulatory regions with GO annotations
        self.fused = dict()
        for c in self.goClasses:
            self.fused[c] = self.geneRegulatory.merge(self.goClasses[c], on=self.gtfGeneCol)
            self.fused[c].rename({4:"GO_Term"}, axis=1, inplace=True)
        # Setup gene-GO matrix for clustering
        self.matrices = dict()
        for c in self.goClasses:
            geneFa, genes = pd.factorize(self.goClasses[c][gtfGeneCol])
            goFa, gos = pd.factorize(self.goClasses[c]["name"])
            data = np.ones_like(goFa, dtype="bool")
            mat = coo_matrix((data, (geneFa, goFa)), shape=(len(genes), len(gos))).toarray().T
            self.matrices[c] = pd.DataFrame(mat)
            self.matrices[c].columns = genes
            self.matrices[c].index = gos
    

    def __getClusters__(self, enrichedGOs, enrichCat):
        pass
    

    def findEnriched(self, query, background=None, clusterize=False):
        enrichs = {}
        for c in self.goClasses:
            regPR = pr.PyRanges(self.fused[c].rename({"name":"Name"}, axis=1))
            if background is not None:
                enrichs[c] = overlap_utils.computeEnrichVsBg(regPR, background, query)
        regPR = pr.PyRanges(self.fused[c].rename({self.gtfGeneCol:"Name"}, axis=1))
        return enrichs




    
