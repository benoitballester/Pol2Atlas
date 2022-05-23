import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyranges as pr
import rpy2.robjects as ro
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from statsmodels.api import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.stats.multitest import fdrcorrection

from .utils import matrix_utils, overlap_utils

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

def customwrap(s,width=20):
    return "<br>".join(textwrap.wrap(s,width=width)).capitalize()

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
    model = GLM(observed, df, family=Binomial())
    model = model.fit([0.0, 0.0], disp=False)
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = (1.0-waldP/2.0)
    return (goTerm, pvals, beta)

def fitBinomModelNoBg(observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([np.ones_like(expected)]).T, 
                                columns=["Intercept"], index=idx)
    model = GLM(observed, df, family=Binomial())
    model = model.fit([0.0], disp=False)
    t_test = model.t_test(f"Intercept = {expected}")
    beta = t_test.z[0]
    waldP = t_test.pvalue[0]
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

    def findEnriched(self, query, background=None, minGenes=3, cores=-1):
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
            expected = intersectBg.loc[obsMatrix.columns]
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        else:
            expected = intersectBg.loc[obsMatrix.columns]
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        endog = pd.merge(observed, expected, right_index = True, left_index = True)
        # Trim GOs under cutoff
        trimmed = obsMatrix[intersectQuery.index[queryGenes]].sum(axis=1) >= minGenes
        # Setup parallel computation settings
        if cores == -1:
            cores = maxCores
        maxBatch = len(obsMatrix[trimmed])
        maxBatch = int(0.25*maxBatch/cores)+1
        hitsPerGO = np.sum(obsMatrix * observed.values.ravel()[None, :], axis=1)
        # Fit a Binomial GLM for each annotation, and evaluate wald test p-value for each gene annotation
        with Parallel(n_jobs=cores, verbose=2, backend="loky", batch_size=maxBatch, max_nbytes=None) as pool:
            if background is not None:
                results = pool(delayed(fitBinomModel)(hasAnnot, endog, expected, gos, queryCounts.index[obsGenes]) for gos, hasAnnot in obsMatrix[trimmed].iterrows())
                # results = [fitBinomModel(hasAnnot, endog, expected, gos, queryCounts.index[obsGenes]) for gos, hasAnnot in obsMatrix[trimmed].iterrows()]
            else:
                # results = pool(delayed(fitBinomModelNoBg)(hasAnnot, endog, expected, gos, queryCounts.index[obsGenes]) for gos, hasAnnot in obsMatrix[trimmed].iterrows())
                results = [fitBinomModelNoBg(hasAnnot, endog, expected, gos, queryCounts.index[obsGenes]) for gos, hasAnnot in obsMatrix[trimmed].iterrows()]
        
        results = pd.DataFrame(results)
        results.set_index(0, inplace=True)
        results.columns = ["P(Beta > 0)", "Beta"]
        results.dropna(inplace=True)
        qvals = results["P(Beta > 0)"].copy()
        qvals.loc[:] = fdrcorrection(qvals)[1]
        results["BH corrected p-value"] = qvals
        results["-log10(qval)"] = -np.log10(qvals)
        results["-log10(pval)"] = -np.log10(results["P(Beta > 0)"])
        results["FC"] = np.exp(results["Beta"])
        results["Name"] = [self.goMap[i] for i in results.index]
        results["Total hits"] = hitsPerGO
        results.sort_values(by="P(Beta > 0)", inplace=True)
        return results

    def plotEnrichs(self, enrichDF, title="", by="P(Beta > 0)", alpha=0.05, topK=10, savePath=None):
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
        ordered = -np.log10(newDF[by][selected]).sort_values(ascending=True)[:topK]
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

    def clusterTreemap(self, enrichDF, alpha=0.05, score="-log10(qval)", resolution=1.0, output=None):
        sig = enrichDF[enrichDF["BH corrected p-value"] < alpha]
        clusters = matrix_utils.graphClustering(csr_matrix(self.mat.loc[sig.index]), 
                                                "dice", k=int(0.5+0.5*np.sqrt(len(sig))), r=resolution, snn=True, 
                                                disconnection_distance=1.0, restarts=10)
        sig["Cluster"] = clusters
        sig["Name"] = [customwrap(self.goMap[i]) for i in sig.index]
        representatives = pd.Series(dict([(i, sig["Name"][sig[score][sig["Cluster"] == i].idxmax()]) for i in np.unique(sig["Cluster"])]))
        sig["Representative"] = representatives[sig["Cluster"]].values
        duplicate = sig["Representative"] == sig["Name"]
        sig.loc[:, "Representative"][duplicate] = ""
        fig = px.treemap(names=sig["Name"], parents=sig["Representative"], 
                        values=sig[score],
                        width=800, height=800)
        fig.update_layout(margin = dict(t=2, l=2, r=2, b=2),
                        font_size=30)
        fig.show()
        if output is not None:
            fig.write_image(output)
            fig.write_html(output + ".html")




    
