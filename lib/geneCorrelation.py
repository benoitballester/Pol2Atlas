import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyranges as pr
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr, pearsonr
from statsmodels.stats.multitest import fdrcorrection
from .utils import matrix_utils, overlap_utils
from . import pyGREATglm
from joblib.externals.loky import get_reusable_executor
maxCores = len(os.sched_getaffinity(0))

class geneCorreler(pyGREATglm.pyGREAT):
    """
    doc
    """
    def fit(self, query, queryNormCounts, geneNormCounts, nPerms=100000):
        """
        Find query-gene associations and compute empirical null distribution of 
        random query-gene correlations.

        Parameters
        ----------
        query: pandas dataframe in bed-like format or PyRanges
            Set of genomic regions to compute enrichment on.
        queryNormCounts: ndarray
            Count table for query regions.
        maxGenes: pandas dataframe (default 3)
            Normalized count table for genes. Rows are features and should match queryNormCounts.
            Columns are genes, column names are gene names.
        nPerms: int, (default 100000)
            Number of random query-gene pairs to evaluate correlation on.
        
        Returns
        -------
        results: pandas dataframe
        """
        # Compute random correlations between normalized counts of query and genes
        randomP2 = np.random.choice(len(queryNormCounts.T), nPerms, replace=True)
        randomGene = np.random.choice(len(geneNormCounts.T), nPerms, replace=True)
        xrank = rankdata(queryNormCounts[:, randomP2], axis=0)
        yrank = rankdata(geneNormCounts.iloc[:, randomGene], axis=0)
        self.randomCorrelations = np.array([pearsonr(xrank[:, i],yrank[:, i])[0] for i in range(yrank.shape[1])])
        geneReg = pr.PyRanges(self.geneRegulatory)
        self.queryPr = query
        self.linkedGene = self.queryPr.join(geneReg).as_df()
        self.queryNormCounts = queryNormCounts
        self.geneNormCounts = geneNormCounts
        return self.linkedGene

    def findCorrelations(self, linkedGenes="fitted", alternative="two-sided"):
        # Compute correlations between normalized counts of query and genes

        if linkedGenes == "fitted":
            linkedGene = self.linkedGene
        else:
            linkedGene = linkedGenes
        self.means = []
        self.corr_name = []
        self.correlations = []
        self.corrP = []
        alt = alternative
        for i in range(len(linkedGene)):
            if (i+1)%100 == 0:
                print(i)
                break
            gene = linkedGene.iloc[i]["gene_name"]
            p2 = linkedGene.iloc[i]["Name"] 
            try:
                exprGene = self.geneNormCounts[gene].values
                if len(exprGene.shape) > 1:
                    print(exprGene.shape, gene)
                    continue
            except KeyError:
                print("Missing", gene)
                continue
            exprP2 = self.queryNormCounts[:, p2]
            self.corr_name.append((p2, gene))
            r, p = spearmanr(exprP2, exprGene)
            if alt == "greater":
                p = (r < self.randomCorrelations[:, 0]).mean()
            elif alt == "less":
                p = (r > self.randomCorrelations[:, 0]).mean()
            elif alt == "two-sided":
                p1 = (r < self.randomCorrelations[:, 0]).mean()
                p2 = (r > self.randomCorrelations[:, 0]).mean()
                p = np.minimum(p1, p2)*2
            self.correlations.append(r)
            self.corrP.append(p)
            self.means.append((np.mean(exprP2),np.mean(exprGene)))
        tab = np.concatenate([np.array(self.corr_name)[:,0], 
                              np.array(self.corr_name)[:,1], 
                              np.array(self.correlations), 
                              np.array(self.corrP)]).reshape(len(self.corrP),-1, order="F")
        return tab

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
        maxBatch = int(0.25*maxBatch/cores) + 1
        hitsPerGO = np.sum(obsMatrix * observed.values.ravel()[None, :], axis=1)
        # Fit a Negative Binomial GLM for each annotation, and evaluate wald test p-value for each gene annotation
        with Parallel(n_jobs=cores, batch_size=maxBatch, max_nbytes=None, mmap_mode=None) as pool:
            results = pool(delayed(pyGREATglm.fitNBinomModel)(hasAnnot, endog, expected, gos, queryCounts.index) for gos, hasAnnot in obsMatrix.loc[trimmed].iterrows())
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
        clusters = matrix_utils.graphClustering(self.mat.loc[sig.index].values.astype(bool), 
                                                metric, k=int(np.sqrt(len(sig))), r=resolution, snn=True, 
                                                approx=False, restarts=10)
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
                                                    approx=False, restarts=10)
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




    
