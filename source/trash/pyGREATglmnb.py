import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import rpy2.robjects as ro
import statsmodels.discrete.discrete_model as discrete_model
from joblib import Parallel, delayed
from rpy2.robjects import FloatVector, StrVector, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.api import GLM
from statsmodels.genmod.families.family import Binomial

from .utils import overlap_utils

revigo = importr("rrvgo")
base = importr("base")
maxCores = len(os.sched_getaffinity(0))

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

def fitBinomModel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    model = discrete_model.Poisson(observed, df, exposure=expected)
    # Set a large alpha as a first guess for better convergence
    model = model.fit([0.0, 0.0], disp=False, maxiter=100)
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = (1.0-waldP/2.0)
    return (goTerm, pvals, beta)

def fitPoimodel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    model = discrete_model.Poisson(observed, df, exposure=expected)
    # Set a large alpha as a first guess for better convergence
    model = model.fit([0.0, 0.0], disp=False, maxiter=100)
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = (1.0-waldP/2.0)
    return (goTerm, pvals, beta)

def fitNBmodel(hasAnnot, observed, expected, goTerm, idx, nbType="nb2", cov_type="nonrobust"):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    model = discrete_model.NegativeBinomial(observed, df, nbType, exposure=expected)
    # Set a large alpha as a first guess for better convergence
    model = model.fit([0.0, 0.0, 10.0], disp=False, maxiter=100, cov_type=cov_type)
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = (1.0-waldP/2.0)
    return (goTerm, pvals, beta)


class pyGREAT:
    """
    doc
    """
    def __init__(self, gmtFile, geneFile, gtfGeneCol="gene_name"):
        self.gtfGeneCol = "gene_name"
        # Parse GMT file
        # Setup gene-GO matrix
        genesPerAnnot = dict()
        self.goMap = dict()
        allGenes = set()
        with open(gmtFile) as f:
            for l in f:
                vals = l.rstrip("\n").split("\t")
                genesPerAnnot[vals[0]] = vals[2:]
                allGenes |= set(vals[2:])
                self.goMap[vals[0]] = vals[1]
        self.mat = pd.DataFrame(columns=allGenes, dtype="int8", index=genesPerAnnot.keys())
        for ann in genesPerAnnot.keys():
            self.mat.loc[ann] = 0
            self.mat.loc[ann][genesPerAnnot[ann]] = 1
        # Read gtf file
        gencode = pr.read_gtf(geneFile)
        gencode = gencode.as_df()
        transcripts = gencode[gencode["Feature"] == "gene"].copy()
        del gencode
        transcripts = transcripts[["Chromosome", "Start", "End", gtfGeneCol, "Strand"]]
        # Reverse positions on opposite strand for convenience
        geneInList = np.isin(list(transcripts[self.gtfGeneCol]), list(allGenes), assume_unique=True)
        reversedTx = transcripts.copy()[["Chromosome", "Start", "End", self.gtfGeneCol]][geneInList]
        reversedTx["Start"] = transcripts["Start"].where(transcripts["Strand"] == "+", transcripts["End"])
        reversedTx["End"] = transcripts["End"].where(transcripts["Strand"] == "+", transcripts["Start"])
        # Apply infered regulatory logic
        self.geneRegulatory = regLogicGREAT(5000, 1000, 1000000)(reversedTx)

    def findEnriched(self, query, background=None, minGenes=2, cores=-1):
        """
        Find enriched terms in genes near query.

        Parameters
        ----------
        query: pandas dataframe in bed-like format or PyRanges
            Set of genomic regions to compute enrichment on.
        background: None, pandas dataframe in bed-like format, or PyRanges (default: None)
            If set to None considers the whole genome as the possible locations of the query.
            Otherwise it supposes the query is a subset of these background regions.
        minGenes: int, (default 3)
            Minimum number of intersected gene for a GO annotation.
        
        Returns
        -------
        results: pandas dataframe or tuple of pandas dataframes
            Three columns pandas dataframe, with for each gene annotation its p-value,
            FDR corrected p-value, and regression coefficient.
        """
        # First compute intersections count for each gene
        # And expected intersection count for each gene
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
        obsGenes = np.isin(queryCounts.index, self.mat.columns)
        queryGenes = np.isin(intersectQuery.index, self.mat.columns)
        obsMatrix = self.mat[queryCounts.index[obsGenes]].copy()
        if background is not None:
            expected = intersectBg.loc[obsMatrix.columns] * len(query) / len(background)
            ratios = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        else:
            expected = intersectBg.loc[obsMatrix.columns]
            ratios = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        # Trim GOs under cutoff
        trimmed = obsMatrix[intersectQuery.index[queryGenes]].sum(axis=1) >= minGenes
        # Setup parallel computation settings
        if cores == -1:
            cores = maxCores
        maxBatch = len(obsMatrix[trimmed])
        maxBatch = int(0.5*maxBatch/cores)+1
        # Fit a NB2 GLM for each annotation, and evaluate wald test p-value for each gene annotation
        # The model is ln(µ) = B0 + B1*G + ln(y)
        # Where G is whether the gene belongs to the annotation set or not, y is the expected value
        # of intersection counts, computed either from the background set or from the gene set coverage
        # µ is the predicted intersection count. B0 and B1 are the regressed coefficient
        # B1 is the tested coefficient
        with Parallel(n_jobs=cores, verbose=2, batch_size=maxBatch) as pool:
            results = pool(delayed(fitPoimodel)(hasAnnot, ratios, expected, gos, queryCounts.index[obsGenes]) for gos, hasAnnot in obsMatrix[trimmed].iterrows())
        results = pd.DataFrame(results)
        results.set_index(0, inplace=True)
        results.columns = ["P(Beta > 0)", "Beta"]
        results.dropna(inplace=True)
        qvals = results["P(Beta > 0)"].copy()
        qvals.loc[:] = fdrcorrection(qvals)[1]
        results["BH corrected p-value"] = qvals
        results.sort_values(by="P(Beta > 0)", inplace=True)
        return results

    def plotEnrichs(self, enrichDF, title="", alpha=0.05, topK=10, savePath=None):
        """
        Draw Enrichment barplots

        Parameters
        ----------
        enrichDF: pandas dataframe or tuple of pandas dataframes
            The result of the findEnriched function
            
        savePath: string (optional)
            If set to None, does not save the figure.
        """
        fig, ax = plt.subplots(figsize=(2,2),dpi=500)
        newDF = enrichDF.copy()
        newDF.index = [self.goMap[i] for i in newDF.index]
        selected = (newDF["BH corrected p-value"] < alpha)
        ordered = -np.log10(1e-250+newDF["BH corrected p-value"][selected]).sort_values(ascending=True)[:topK]
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

    def revigoTreemap(self, enrichDF, output=None, simplification=0.9):
        gos = enrichDF.index
        scores = -np.log10(1e-250+enrichDF["P(Beta > 0)"][enrichDF["BH corrected p-value"] < 0.05])
        scores.index = gos[enrichDF["BH corrected p-value"] < 0.05]
        numpy2ri.deactivate()
        simMatrix = revigo.calculateSimMatrix(StrVector(scores.index),
                                            orgdb="org.Hs.eg.db",
                                            ont="BP",
                                            method="Rel")
        v = FloatVector(scores.values)
        v.names = list(scores.index)
        reducedTerms = revigo.reduceSimMatrix(simMatrix,
                                        v,
                                        threshold=simplification,
                                        orgdb="org.Hs.eg.db")
        numpy2ri.deactivate()
        grdevices = importr('grDevices')
        grdevices.pdf(file=output, width=5, height=5)
        revigo.treemapPlot(reducedTerms)
        # plotting code here
        grdevices.dev_off()
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            return pd.DataFrame(reducedTerms)
                




    
