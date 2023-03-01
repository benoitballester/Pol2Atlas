import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyranges as pr
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from statsmodels.api import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.api import NegativeBinomial, Poisson
from statsmodels.stats.multitest import fdrcorrection
from sklearn.cluster import AgglomerativeClustering
from .utils import matrix_utils, overlap_utils
from joblib.externals.loky import get_reusable_executor

maxCores = len(os.sched_getaffinity(0))

class regLogicGREAT:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF, chrInfo):
        # Infered regulatory domain logic
        copyTx = txDF.copy()
        copyTx["Start"] = (txDF["Start"] - self.upstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] - self.downstream)
        copyTx["End"] = (txDF["Start"] + self.downstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] + self.upstream)
        copyTx.sort_values(["Chromosome", "Start"], inplace=True)
        try:
            copyTx["Chromosome"].cat.remove_unused_categories(inplace=True)
        except:
            pass
        gb = copyTx.groupby("Chromosome")
        copyTx["Chromosome"] = copyTx["Chromosome"]
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = copyTx["Chromosome"] == c
            nextReg = np.roll(copyTx["Start"][inIdx], -1)
            try:
                nextReg[-1] = chrInfo.loc[c].values[0]
            except:
                print(f"Warning: chromosome '{c}' in gtf but not in size file, skipping all genes within this chromosome.")
                copyTx = copyTx[np.logical_not(inIdx)]
                continue
            previousReg = np.roll(copyTx["End"][inIdx], 1)
            previousReg[0] = 0
            extMin = np.maximum(copyTx["Start"][inIdx] - self.distal, previousReg)
            extMax = np.minimum(copyTx["End"][inIdx] + self.distal, nextReg)
            extMin = np.minimum(copyTx["Start"][inIdx], extMin)
            extMax = np.maximum(copyTx["End"][inIdx], extMax)
            copyTx.loc[copyTx["Chromosome"] == c, "Start"] = np.clip(extMin, 0, chrInfo.loc[c].values[0])
            copyTx.loc[copyTx["Chromosome"] == c, "End"] = np.clip(extMax, 0, chrInfo.loc[c].values[0])
        return copyTx

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

def fitNBinomModel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    model = NegativeBinomial(observed, df, exposure=expected, loglike_method="nb1")
    model = model.fit([0.0,0.0,10.0], method="lbfgs", disp=False)
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
    def __init__(self, gmtFile, geneFile, chrFile, validGenes="all", 
                 distal=1000000, upstream=5000, downstream=1000):
        self.chrInfo = pd.read_csv(chrFile, sep="\t", index_col=0, header=None)
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
        # Read gtf file
        gencode = pr.read_gtf(geneFile)
        gencode = gencode.as_df()
        self.transcripts = gencode[gencode["Feature"] == "gene"].copy()
        del gencode
        self.transcripts = self.transcripts[["Chromosome", "Start", "End", self.gtfGeneCol, "Strand"]]
        # Reverse positions on opposite strand for convenience
        reversedTx = self.transcripts.copy()[["Chromosome", "Start", "End", self.gtfGeneCol, "Strand"]]
        self.txList = reversedTx.copy()
        self.clusters = None
        # Apply infered regulatory logic
        self.geneRegulatory = regLogicGREAT(upstream, downstream, distal)(reversedTx, self.chrInfo)
        # geneInList = np.isin(list(self.geneRegulatory["gene_name"]), list(allGenes), assume_unique=False)
        self.geneRegulatory.drop("Strand", 1, inplace=True)
        self.geneRegulatory.index = self.geneRegulatory["gene_name"]
        self.geneRegulatory = self.geneRegulatory[~self.geneRegulatory.index.duplicated(False)]
        if validGenes == "annotated":
            validGenes = pd.Index(self.geneRegulatory["gene_name"]).intersection(allGenes)
        elif validGenes == "all":
            validGenes = self.geneRegulatory["gene_name"]
        else:
            print("Invalid validGenes argument", validGenes)
            print("Use either 'annotated' or 'all'")
            return None
        self.mat = pd.DataFrame(columns=validGenes, dtype="int8", index=genesPerAnnot.keys())
        for ann in genesPerAnnot.keys():
            self.mat.loc[ann] = 0
            try:
                self.mat.loc[ann][self.mat.columns.intersection(genesPerAnnot[ann])] = 1
            except KeyError:
                print("Missing", genesPerAnnot[ann])
                continue
        self.geneRegulatory = self.geneRegulatory.loc[self.mat.columns]

    def findEnriched(self, query, background=None, minGenes=3, maxGenes=1000, cores=-1):
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
            Minimum number of intersected genes for a GO annotation.
        maxGenes: int, (default 3)
            Maximum number of genes for a GO annotation.
        cores: int, (default -1)
            Max number of cores to used for parallelized computations. Default uses all
            available cores (-1).
        
        Returns
        -------
        results: pandas dataframe
        """
        # First compute intersections count for each gene
        # And expected intersection count for each gene
        regPR = pr.PyRanges(self.geneRegulatory.rename({self.gtfGeneCol:"Name"}, axis=1))
        if background is not None:
            intersectBg = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(background, useSummit=False))
        else:
            genomeSize = np.sum(self.chrInfo).values[0]
            intersectBg = (self.geneRegulatory["End"]-self.geneRegulatory["Start"])/genomeSize
            intersectBg = np.maximum(intersectBg, 1/genomeSize)
            intersectBg.index = self.geneRegulatory["gene_name"]
            intersectBg = intersectBg.groupby(intersectBg.index).sum()
        intersectQuery = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(query, useSummit=False))
        queryCounts = intersectBg.copy() * 0
        queryCounts.loc[intersectQuery.index] = intersectQuery
        obsMatrix = self.mat[queryCounts.index].copy()
        if background is not None:
            expected = intersectBg.loc[obsMatrix.columns]
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index])
            endog = observed.copy()
            expected *= len(query)/len(background)
        else: 
            expected = intersectBg.loc[obsMatrix.columns]*len(query)
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index])
            endog = observed.copy()
        # Trim GOs under cutoff
        trimmed = obsMatrix[intersectQuery.index].sum(axis=1) >= minGenes
        trimmed = trimmed & (obsMatrix.sum(axis=1) <= maxGenes)
        # Setup parallel computation settings
        if cores == -1:
            cores = maxCores      
        maxBatch = len(obsMatrix.loc[trimmed])
        maxBatch = int(0.25*maxBatch/cores)+1
        hitsPerGO = np.sum(obsMatrix * observed.values.ravel()[None, :], axis=1)
        # Fit a Negative Binomial GLM for each annotation, and evaluate wald test p-value for each gene annotation
        with Parallel(n_jobs=cores, batch_size=maxBatch, max_nbytes=None, mmap_mode=None) as pool:
            results = pool(delayed(fitNBinomModel)(hasAnnot, endog, expected, gos, queryCounts.index) for gos, hasAnnot in obsMatrix.loc[trimmed].iterrows())
        # Manually kill workers afterwards or they'll just stack up with multiple runs
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
        # Format results
        results = pd.DataFrame(results)
        results.set_index(0, inplace=True)
        results.columns = ["P(Beta > 0)", "Beta"]
        results.dropna(inplace=True)
        results["P(Beta > 0)"] = np.maximum(results["P(Beta > 0)"], 1e-320)
        qvals = results["P(Beta > 0)"].copy()
        qvals.loc[:] = fdrcorrection(qvals.values)[1]
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

    def clusterTreemap(self, enrichDF, alpha=0.05, score="-log10(qval)", metric="yule", resolution=1.0, output=None):
        sig = enrichDF[enrichDF["BH corrected p-value"] < alpha]
        simplifiedMat = self.mat.loc[sig.index].values.astype(bool)
        clusters = matrix_utils.graphClustering(simplifiedMat, 
                                                metric, k=int(np.sqrt(len(sig))), r=resolution, snn=True, 
                                                approx=True, restarts=10)
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

    def clusterTreemapFull(self, enrichDF, alpha=0.05, score="-log10(qval)", metric="yule", resolution=1.0, output=None):
        sig = enrichDF[enrichDF["BH corrected p-value"] < alpha]
        if self.clusters is None:
            self.clusters = matrix_utils.graphClustering(self.mat.loc[sig.index].values.astype(bool), 
                                                    metric, k=int(np.sqrt(len(sig))), r=resolution, snn=True, 
                                                    approx=True, restarts=10)
        sig["Cluster"] = self.clusters[np.isin(self.mat.index, sig.index)]
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




    
