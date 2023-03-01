from statistics import median
import warnings
import os
import kneed
import KDEpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import statsmodels.discrete.discrete_model as discrete_model
import rpy2.robjects as ro
from joblib import Parallel, delayed
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import chi2, rankdata, mannwhitneyu, gmean, nbinom, norm, shapiro
from sklearn.decomposition import PCA, TruncatedSVD
from statsmodels.stats.multitest import fdrcorrection
from joblib.externals.loky import get_reusable_executor
from statsmodels.api import GLM
from statsmodels.genmod.families.family import NegativeBinomial
from sklearn.preprocessing import StandardScaler
import pandas as pd
import subprocess
from settings import params, paths
from scipy.sparse import csr_array
from scipy.io import mmread, mmwrite
scran = importr("scran")
deseq = importr("DESeq2")

def findMode(arr):
    # Finds the modal value of a continuous sample
    return np.percentile(arr, 50)
    pos, fitted = KDEpy.FFTKDE(bw="silverman").fit(arr).evaluate(100000)
    return pos[np.argmax(fitted)]


def statsProcess(alpha, sf, counts, design):
    alpha = np.clip(alpha, 1e-5, 1e5)
    pred = np.mean(counts/sf)*sf
    distrib = NegativeBinomial(alpha=alpha)
    pearson = distrib.resid_dev(counts, pred)
    pearson -= np.mean(pearson)
    chi2p = chi2(len(sf)-design.shape[1]).sf(np.sum(np.square(pearson), axis=0))
    return pearson.astype("float32"), chi2p 

def fitModels(counts, sfs, m, design):
    warnings.filterwarnings("error")
    try:
        model = discrete_model.NegativeBinomial(counts.reshape(-1,1), design, exposure=sfs)
        fit = model.fit([np.log(m)]+ [10.0], method="nm", ftol=1e-9, maxiter=500, disp=False, skip_hessian=True)
    except:
        return -1
    warnings.filterwarnings("default")
    return fit.params[-1]

class RnaSeqModeler:
    '''
    doc
    '''
    def __init__(self):
        '''
        doc
        '''
        pass
    
    def fit(self, counts, sf, design=None, maxThreads=-1, subSampleEst=5000, plot=True, verbose=True,
            figSaveDir=None):
        '''
        Fit the model
        '''
        if design is None:
            design = np.ones([len(sf),1], dtype="float32")
        # Setup size factors
        self.counts = counts
        self.scaled_sf = (sf/np.mean(sf)).astype("float32")
        self.normed = (counts / self.scaled_sf.reshape(-1,1)).astype("float32")
        # Estimate Negative Binomial parameters per Pol II probe
        fittedParams = []
        np.random.seed(42)
        subSampleEst = np.minimum(subSampleEst, counts.shape[1])
        if subSampleEst == -1:
            subSampleEst = counts.shape[1]
        shuffled = np.random.permutation(counts.shape[1])[:subSampleEst]
        m = np.mean(self.normed, axis=0)
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=512, max_nbytes=None) as pool:
           fittedParams = pool(delayed(fitModels)(self.counts[:, i], self.scaled_sf, m[i], design) for i in shuffled)
        valid = np.array(fittedParams) > 0.0
        alphas = np.maximum(1e-9, np.array(fittedParams))
        # Estimate NB overdispersion in function of mean expression
        # Overdispersion can be caused by biological variation as well as technical variation
        # To estimate technical overdispersion, it is assumed that a large fraction of probes are non-DE, and that overdispersion is a function of mean
        # The modal value of overdispersion is tracked for each decile of mean expression using a kernel density estimate
        # The median value would overestimate overdispersion as the % of DE probes is unknown
        means = np.mean(self.normed[:, shuffled], axis=0)
        nQuantiles = 25
        pcts = np.linspace(0,100,nQuantiles+1)
        centers = (pcts * 0.5 + np.roll(pcts,1)*0.5)[1:]/100
        quantiles = np.percentile(means, pcts)
        quantiles[0] = 0
        digitized = np.digitize(means, quantiles)
        regressed = []
        for i in np.arange(1,nQuantiles+1):
            if (valid & (digitized == i)).sum() > 5:
                regressed.append(findMode(alphas[valid & (digitized == i)]))
            else:
                regressed.append(1e-5)
        regressed = np.array(regressed).ravel()
        """for i in np.arange(len(regressed)-1)[::-1]:
            regressed[i] = np.maximum(regressed[i], regressed[i+1])"""
        self.means = np.mean(self.normed, axis=0)
        fittedAlpha = si.interp1d(centers, regressed, bounds_error=False, fill_value=(regressed[0], regressed[-1]))
        self.regAlpha = fittedAlpha((rankdata(self.means)-0.5)/len(self.means))
        # Dispatch accross multiple processes
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=512, max_nbytes=None) as pool:
            stats = pool(delayed(statsProcess)(self.regAlpha[i], self.scaled_sf, counts[:, i], design) for i in range(counts.shape[1]))
        # stats = [statsProcess(self.regAlpha[i], self.scaled_sf, counts[:, i], self.means[i]) for i in range(counts.shape[1])]
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
        # Unpack results
        self.residuals = np.array([i[0] for i in stats]).T
        self.pvals = np.array([i[1] for i in stats]).ravel()
        self.hv = fdrcorrection(self.pvals)[0]
        if plot:
            plt.figure(dpi=500)
            plt.plot(self.regAlpha[np.argsort(np.mean(self.normed, axis=0))])
            plt.scatter(np.argsort(np.argsort(np.mean(self.normed[:, shuffled], axis=0)))*self.normed.shape[1]/len(alphas), alphas, s=0.5, linewidths=0, c="red")
            # plt.yscale("log")
            plt.xlabel("Pol II ranked mean expression")
            plt.ylim(-1e-2, 1e2)
            plt.ylabel("Alpha (overdispersion)")
            plt.show()
            if figSaveDir is not None:
                plt.savefig(figSaveDir + "/alpha_trendline.pdf")
            plt.close()
            # Plot mean/variance relationship and selected probes
            v = np.var(self.normed[:, :self.normed.shape[1]], axis=0)
            m = np.mean(self.normed[:, :self.normed.shape[1]], axis=0)
            c = np.array([[0.0,0.0,1.0]]*(self.normed.shape[1]))
            c[self.hv] = [1.0,0.0,0.0]
            plt.figure(dpi=500)
            plt.scatter(m, v, s = 0.5*(100000/len(m)), linewidths=0, c=c, alpha=0.1)
            plt.scatter(m, m+m*m*self.regAlpha, s = 1.0, linewidths=0, c=[0.0,1.0,0.0])
            pts = np.geomspace(m.min(), m.max())
            plt.plot(pts, pts)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Pol II probe mean")
            plt.ylabel("Pol II probe variance")
            if figSaveDir is not None:
                plt.savefig(figSaveDir + "/mv_trendline.pdf")
            plt.show()
            plt.close()
        return self

        

def permutationPA_PCA(X, perm=3, alpha=0.01, solver="randomized", whiten=False,
                      max_rank=None, mincomp=0, returnModel=False, plot=True, figSaveDir=None):
    """
    Permutation Parallel Analysis to find the optimal number of PCA components.

    Parameters
    ----------
    X: ndarray
        obs, features matrix to compute PCA on

    perm: int (default 3)
        Number of permutations. On large matrices the eigenvalues are 
        very stable, so there is no need to use a large number of permutations,
        and only one permutation can be a reasonable choice on large matrices.
        On smaller matrices the number of permutations should be increased.

    alpha: float (default 0.01)
        Permutation p-value threshold.
    
    solver: "arpack" or "randomized"
        Chooses the SVD solver. Randomized is faster but less accurate.
    
    whiten: bool (default True)
        If set to true, each component is transformed to have unit variance.

    max_rank: int or None (default None)
        Maximum number of principal components to compute. Must be strictly less 
        than the minimum of n_features and n_samples. If set to None, computes
        up to min(n_samples, n_features)-1 components.
    
    mincomp: int (default 0)
        Number of components to return

    returnModel: bool (default None)
        Whether to return the fitted PCA model or not. (The full model computed up to max_rank)

    plot: bool (default None)
        Whether to plot the eigenvalues of not

    Returns
    -------
    decomp: ndarray of shape (n obs, k components)
        PCA decomposition with optimal number of components

    model: sklearn PCA object
        Returned only if returnModel is set to true
    """
    # Compute eigenvalues of observed data
    ref = PCA(max_rank, whiten=whiten, svd_solver=solver, random_state=42)
    decompRef = ref.fit_transform(X)
    dstat_obs = ref.explained_variance_
    # Compute permutation eigenvalues
    dstat_null = np.zeros((perm, len(dstat_obs)))
    np.random.seed(42)
    for b in range(perm):
        X_perm = np.apply_along_axis(np.random.permutation, 0, X)
        perm = PCA(max_rank, whiten=whiten, svd_solver=solver, random_state=42)
        perm.fit(X_perm)
        dstat_null[b, :] = perm.explained_variance_

    # Compute p values
    pvals = np.ones(len(dstat_obs))
    delta = np.zeros(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = np.mean(dstat_null[:, i] >= dstat_obs[i])
        delta[i] = dstat_obs[i] /np.mean(dstat_null[:, i])
    for i in range(1, len(dstat_obs)):
        pvals[i] = 1.0-(1.0-pvals[i - 1])*(1.0-pvals[i])
     
    # estimate rank
    r_est = max(sum(pvals <= alpha),mincomp)
    if r_est == max_rank:
        print("""WARNING, estimated number of components is equal to maximal number of computed components !\n Try to rerun with higher max_rank.""")
    if plot:
        plt.figure(dpi=500)
        plt.plot(np.arange(len(dstat_obs))+1, dstat_obs)
        for i in range(len(dstat_null)):
            plt.plot(np.arange(len(dstat_obs))+1, dstat_null[i], linewidth=0.2)
        plt.xlabel("PCA rank")
        plt.ylabel("Eigenvalues / Explained variance (log scale)")
        plt.yscale("log")
        plt.legend(["Observed eigenvalues"])
        plt.xlim(1,r_est*1.2)
        plt.ylim(np.min(dstat_null[:, :int(r_est*1.2)])*0.95, dstat_obs.max()*1.05)
        if figSaveDir is not None:
            plt.savefig(figSaveDir + "/PCA_PA.pdf")
        plt.show()
    if plot:
        plt.figure(dpi=500)
        plt.plot(np.arange(len(pvals))+1, pvals)
        plt.xlabel("PCA rank")
        plt.ylabel("p-value")
        plt.xlim(1,r_est*1.2)
        plt.show()
    if returnModel:
        return decompRef[:, :r_est], ref
    else:
        return decompRef[:, :r_est]


def runScript(script, argumentList, outFile=None):
    # Runs the command as a standard bash command
    # script is command name without path
    # argumentList is the list of argument that will be passed to the command
    if outFile == None:
        subprocess.run([script] + argumentList)
    else:
        with open(outFile, "wb") as outdir:
            subprocess.run([script] + argumentList, stdout=outdir)

def saveDataset(counts, annot, path):
    """ tempCountPath = prefix + "_counts.mtx"
    tempNames = prefix + "indexes.txt"
    tempLabels = prefix + "LabelsDE.txt" """
    mmwrite(path + "counts.mtx", csr_array(counts))
    annot.to_csv(path + "samples.csv", header=None, index=None)
    


def limma1vsAll(counts, sf, annot, probeNames, deFolder):
    hashVal = hash(np.mean(counts))
    tempCountPath = paths.tempDir + str(hashVal) + "CountsDE.mtx"
    tempNames = paths.tempDir + str(hashVal) + "NzIdxDE.txt"
    tempSf = paths.tempDir + str(hashVal) + "SfDE.txt"
    tempLabels = paths.tempDir + str(hashVal) + "LabelsDE.txt"
    mmwrite(tempCountPath, csr_array(counts))
    np.savetxt(tempNames, probeNames, fmt="%i")
    np.savetxt(tempSf, sf)
    annot.to_csv(tempLabels)
    runScript("Rscript", ["--vanilla", "lib/limma.r", tempCountPath, tempLabels, tempNames,
                     tempSf, deFolder])


def filterDetectableGenes(counts, readMin, expMin):
    return np.sum(counts >= readMin, axis=0) >= expMin

def scranNorm(counts):
    detected = [np.sum(counts > i, axis=0) for i in range(5)][::-1]
    mostDetected = np.lexsort(detected)[::-1][:int(counts.shape[1]*0.05+1)]
    with localconverter(ro.default_converter + numpy2ri.converter):
        sf = scran.calculateSumFactors(counts.T[mostDetected])
    return sf

def topFpkmNorm(counts):
    detected = [np.sum(counts > i, axis=0) for i in range(5)][::-1]
    mostDetected = np.lexsort(detected)[::-1][:int(counts.shape[1]*0.05+1)]
    sf = np.sum(counts[:, mostDetected], axis=1)
    return sf / np.mean(sf)

def deseqNorm(counts):
    gmeans = gmean(counts, axis=0)
    gmeans = np.where(counts.min(axis=0) == 0, np.nan, gmeans)
    return np.nanmedian(counts/gmeans, axis=1)
    

def deseqDE(counts, sf, labels, colNames, test="Wald", parallel=False):
    countTable = pd.DataFrame(counts.T, columns=colNames)
    infos = pd.DataFrame(labels, index=colNames, columns=["Type"])
    infos["sizeFactor"] = sf.ravel()
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula("~Type"))
        dds = deseq.DESeq(dds, test=test, fitType="local", parallel=parallel)
        res = deseq.results(dds)
    res = pd.DataFrame(res.slots["listData"], index=res.slots["listData"].names).T
    res["padj"] = np.nan_to_num(res["padj"], nan=1.0)
    return res
""" 
def deseqDE1vsAll(counts, sf, labels, colNames, test="Wald", parallel=False):
    countTable = pd.DataFrame(counts.T, columns=colNames)
    infos = pd.DataFrame(labels, index=colNames, columns=["Type"])
    infos["sizeFactor"] = sf.ravel()
    allresults = dict()
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula("~0 + Type"))
        dds = deseq.DESeq(dds, fitType="local", parallel=parallel)
        print(list(deseq.resultsNames(dds)))
    for i in list(deseq.resultsNames(dds)):
        print(i)
        res = deseq.results(dds, name=i)
        res = pd.DataFrame(res.slots["listData"], index=res.slots["listData"].names).T
        res["padj"] = np.nan_to_num(res["padj"], nan=1.0)
        allresults[i] = res
    return allresults """



def mannWhitneyDE(vals, labels):
    countsC1 = vals[labels == 0]
    countsC2 = vals[labels == 1]
    delta = np.mean(countsC2, axis=0) - np.mean(countsC1, axis=0)
    return mannwhitneyu(countsC1, countsC2)[1], delta