import pandas as pd
import numpy as np
import pyranges as pr
import statsmodels.discrete.discrete_model as discrete_model
from statsmodels.genmod.families.family import Binomial, Poisson
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy.sparse import coo_matrix, csr_matrix
from .utils import overlap_utils, matrix_utils
from scipy.stats import  t
import matplotlib.pyplot as plt
import warnings


class regLogicGREAT:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF):
        # Infered regulatory domain logic
        txDF.sort_values(by=["Chromosome", "Start"], inplace=True)
        regPm = txDF["Start"] - self.upstream * np.sign(txDF["End"]-txDF["Start"])
        regPp = txDF["Start"] + self.downstream * np.sign(txDF["End"]-txDF["Start"])
        gb = txDF.groupby("Chromosome")
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = txDF["Chromosome"] == c
            previousReg = np.roll(regPp[inIdx], 1)
            previousReg[0] = 0
            previousReg[-1] = int(1e10)
            nextReg = np.roll(regPm[inIdx], -1)
            nextReg[-1] = int(1e10)
            extendedM = np.maximum(txDF["Start"][inIdx] - self.distal, np.minimum(previousReg, regPm[inIdx]))
            extendedP = np.minimum(txDF["Start"][inIdx] + self.distal, np.maximum(nextReg, regPp[inIdx]))
            txDF.loc[txDF["Chromosome"] == c, "Start"] = extendedM
            txDF.loc[txDF["Chromosome"] == c, "End"] = extendedP
        return txDF


def capTxtLen(txt, maxlen):
    try:
        if len(txt) < maxlen:
            return txt
        else:
            return txt[:maxlen] + '...'
    except:
        return "N/A"

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
        print("a")
        transcripts.to_csv("gencode_tx.bed", sep="\t", header=None, index=None)
        # Reverse positions on opposite strand for convenience
        geneInList = np.isin(transcripts[self.gtfGeneCol], np.unique(goAnnotation[2]), assume_unique=True)
        reversedTx = transcripts.copy()[["Chromosome", "Start", "End", self.gtfGeneCol]][geneInList]
        reversedTx["Start"] = transcripts["Start"].where(transcripts["Strand"] == "+", transcripts["End"])
        reversedTx["End"] = transcripts["End"].where(transcripts["Strand"] == "+", transcripts["Start"])
        self.geneRegulatory = regLogicGREAT(5000, 1000, 1000000)(reversedTx)
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
    

    def findEnriched(self, query, background=None, clusterize=False, minGenes=3):
        """
        Find enriched terms in genes near query.

        Parameters
        ----------
        query: pandas dataframe in bed-like format or PyRanges
            Set of genomic regions to compute enrichment on.
        background: None, pandas dataframe in bed-like format, or PyRanges (default: None)
            If set to None considers the whole genome as the possible locations of the query.
            Otherwise it supposes the query is a subset of these background regions.
        clusterize: bool (default: False)
            If set to True, groups gene annotations that appears in similar genes to avoid
            redundancy in the annotation.
        
        Returns
        -------
        results: pandas dataframe or tuple of pandas dataframes
            Three columns pandas dataframe, with for each gene annotation its p-value,
            FDR corrected p-value, and regression coefficient.
            If clusterize was set to True, returns a dataframe for each cluster, stored inside a tuple.
        """
        enrichs = {}
        regPR = pr.PyRanges(self.geneRegulatory.rename({self.gtfGeneCol:"Name"}, axis=1))
        if background is not None:
            intersectBg = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(background, useSummit=False))
        else:
            intersectBg = (self.geneRegulatory["End"]-self.geneRegulatory["Start"])/3e9 * len(query)
            intersectBg = np.maximum(intersectBg, 1/(3e9))
            intersectBg.index = self.geneRegulatory["gene_name"]
            intersectBg = intersectBg.groupby(intersectBg.index).sum()
        intersectQuery = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(query, useSummit=False))
        queryCounts = intersectBg.copy() * 0
        queryCounts.loc[intersectQuery.index] = intersectQuery
        obsGenes = np.isin(queryCounts.index, self.matrices["biological_process"].columns)
        queryGenes = np.isin(intersectQuery.index, self.matrices["biological_process"].columns)
        obsMatrix = self.matrices["biological_process"][queryCounts.index[obsGenes]]
        if background is not None:
            expected = intersectBg.loc[obsMatrix.columns]
            ratios = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        else:
            expected = intersectBg.loc[obsMatrix.columns]
            ratios = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        pvals = pd.Series()
        beta = pd.Series()
        for gos, hasAnnot in (obsMatrix.iterrows()):
            if hasAnnot.loc[intersectQuery.index[queryGenes]].sum() >= minGenes:
                df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=queryCounts.index[obsGenes])
                # For some reason only the regularized solver does not have a lot of convergence issues
                model = discrete_model.NegativeBinomial(ratios, df, "nb1", exposure=expected).fit_regularized(disp=0, method="l1", alpha=0.0, trim_mode="off")
                beta[gos] = model.params["GS"]
                # Get one sided pvalues
                if model.params["GS"] >= 0:
                    pvals[gos] = model.pvalues["GS"]/2.0
                else:
                    pvals[gos] = (1.0-model.pvalues["GS"]/2.0)
        qvals = pvals.copy()
        qvals.dropna(inplace=True)
        qvals.loc[:] = fdrcorrection(qvals)[1]
        results = pd.concat([pvals, qvals, beta], axis=1)
        results.columns = ["P(Beta > 0)", "BH corrected p-value", "Beta"]
        results.sort_values(by="P(Beta > 0)", inplace=True)
        if clusterize:
            mat = self.matrices["biological_process"]
            grps = matrix_utils.graphClustering(mat.values.astype(bool), "dice", 30, r=900.0, snn=True, restarts=10, disconnection_distance=1.0)
            dfs = []
            obsAnnots = np.isin(mat.index, results.index)
            subClusts = grps[obsAnnots]
            subMat = mat.index[obsAnnots]
            for i in np.unique(subClusts):
                idx = subMat[i == subClusts]
                dfs.append(results.loc[idx])
            results = tuple(dfs)
        return results

    def plotEnrichs(self, enrichDF, title="", alpha=0.05, topK=10, savePath=None):
        """
        Draw Enrichment barplots

        Parameters
        ----------
        enrichDF: pandas dataframe or tuple of pandas dataframes
            
        savePath: string (optional)
            If set to None, does not save the figure.
        """
        fig, ax = plt.subplots(figsize=(2,2),dpi=500)
        if type(enrichDF) is tuple:
            newDF = pd.DataFrame(columns=["P(Beta > 0)", "BH corrected p-value", "Beta"])
            for df in enrichDF:
                imax = df["P(Beta > 0)"].idxmax(axis=0)
                newDF.loc[imax] = df.loc[imax]
        else:
            newDF = enrichDF
        selected = (newDF["BH corrected p-value"] < alpha)
        ordered = -np.log10(newDF["BH corrected p-value"][selected]).sort_values(ascending=True)[:topK]
        terms = ordered.index
        t = [capTxtLen(term, 50) for term in terms]
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(length=3, width=1.2)
        ax.barh(range(len(terms)), np.minimum(ordered[::-1],324.0))
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(t[::-1], fontsize=5)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("-log10(Corrected P-value)", fontsize=8)
        ax.set_title(title, fontsize=10)
        if savePath is not None:
            fig.savefig(savePath, bbox_inches="tight")
        return fig, ax

        




    
