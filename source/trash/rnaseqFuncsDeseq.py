import numpy as np
import statsmodels.discrete.discrete_model as discrete_model
from statsmodels.genmod.families import family
import warnings
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.special import erfinv
from statsmodels.stats.multitest import fdrcorrection
import scipy.interpolate as si
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.preprocessing import StandardScaler
import KDEpy
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata
import scipy.interpolate as si
from joblib import Parallel, delayed
import scipy.stats as ss
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
scran = importr("scran")
deseq = importr("DESeq2")
base = importr("base")
s4 = importr("S4Vectors")

def findMode(arr):
    # Finds the modal value of a continuous sample
    # Scale to unit variance for numerical stability
    scaler = RobustScaler()
    values = scaler.fit_transform(arr.reshape(-1,1)).ravel()
    pos, fitted = KDEpy.FFTKDE(bw="silverman", kernel="gaussian").fit(values).evaluate(100000)
    return scaler.inverse_transform(pos[np.argmax(fitted)].reshape(-1,1)).ravel()


def statsProcess(regAlpha,scaled_ni,countsSel,means):
    # 
    func = family.NegativeBinomial(alpha=regAlpha)
    pred = np.repeat(means, len(countsSel)) * scaled_ni
    anscombeResiduals = func.resid_anscombe(countsSel, pred)
    dev_raw = func.deviance(countsSel, pred)
    res_dev = func.resid_dev(countsSel, pred)
    return anscombeResiduals, dev_raw, res_dev


def nullDeviance(mean, alpha, sf, nSamples = 10000):
    nbMean = mean * sf
    alpha = alpha
    n = 1 / alpha
    p = nbMean / (nbMean + alpha * (nbMean**2))
    devs = np.zeros(nSamples)
    rv = ss.nbinom.rvs(n,p,size=(nSamples,len(p)))
    func = family.NegativeBinomial(alpha=alpha)
    resid = func.resid_anscombe(rv, nbMean)
    devs = np.sum(np.square(resid), axis=1)
    return devs


class RnaSeqModeler:
    '''
    doc
    '''
    def __init__(self):
        '''
        doc
        '''
        pass
    
    def fit(self, counts, sf, subSampleEst=20000, plot=True, verbose=True):
        '''
        Fit the model
        '''
        # Setup size factors
        self.scaled_sf = sf/np.mean(sf)
        self.normed = counts / self.scaled_sf.reshape(-1,1)
        # Estimate Negative Binomial parameters per Pol II probe
        fittedParams = []
        worked = []
        np.random.seed(42)
        subSampleEst = np.minimum(subSampleEst, counts.shape[1])
        shuffled = np.random.permutation(counts.shape[1])[:subSampleEst]
        countTable = pd.DataFrame(counts.T[shuffled], columns=np.arange(len(counts)))
        infos = pd.DataFrame(np.arange(len(counts)), index=np.arange(len(counts)), columns=["Type"])
        infos["sizeFactor"] = sf.ravel()
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            self.dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula("~1"))
            self.dds  = deseq.estimateDispersions_DESeqDataSet(self.dds, fitType="local")
        worked = shuffled
        dollar = base.__dict__["$"]
        alphas = np.array(dollar(s4.mcols(self.dds), "dispMAP"))
        # Estimate NB overdispersion in function of mean expression
        # Overdispersion can be caused by biological variation as well as technical variation
        # To estimate technical overdispersion, it is assumed that a large fraction of probes are non-DE, and that overdispersion is a function of mean
        # The modal value of overdispersion is tracked for each decile of mean expression using a kernel density estimate
        # The median value would overestimate overdispersion as the % of DE probes is unknown
        means = np.mean(self.normed[:, worked], axis=0)
        nQuantiles = 10
        pcts = np.linspace(0,100,nQuantiles+1)
        centers = (pcts * 0.5 + np.roll(pcts,1)*0.5)[1:]/100
        quantiles = np.percentile(means, pcts)
        quantiles[0] = 0
        digitized = np.digitize(means, quantiles)
        regressed = []
        for i in np.arange(1,nQuantiles+1):
            regressed.append(findMode(alphas[digitized==i]))
        regressed = np.array(regressed).ravel()
        self.means = np.mean(self.normed, axis=0)
        fittedAlpha = si.interp1d(centers, regressed, bounds_error=False, fill_value="extrapolate")
        self.regAlpha = fittedAlpha((rankdata(self.means)-0.5)/len(self.means))
        if plot:
            plt.figure(dpi=500)
            plt.plot(self.regAlpha[np.argsort(np.mean(self.normed, axis=0))])
            plt.scatter(np.argsort(np.argsort(np.mean(self.normed[:, worked], axis=0)))*self.normed.shape[1]/len(alphas), alphas, s=0.5, linewidths=0, c="red")
            plt.yscale("log")
            plt.xlabel("Pol II ranked mean expression")
            plt.ylabel("Alpha (overdispersion)")
        # Dispatch accross multiple processes
        with Parallel(n_jobs=-1, verbose=verbose, batch_size=512) as pool:
            stats = pool(delayed(statsProcess)(self.regAlpha[i], self.scaled_sf, counts[:, i], self.means[i]) for i in range(counts.shape[1]))
        # Unpack results
        self.anscombeResiduals = np.array([i[0] for i in stats]).T
        self.deviances = np.sum(self.anscombeResiduals**2, axis=0)
        self.res_dev = np.array([i[2] for i in stats]).T
        return self

    
    def hv_selection(self, evalsPts=100, alpha=0.05, maxOutlierDeviance=0.75, plot=True):
        '''
        Select features not not well modelized by the fitted NB model.
        Requires to have run the fit method before.

        Parameters
        ----------
        evalsPts: int (default 100)
            The number of points at which to evaluate the CDF of the deviance

        alpha: float (default 0.05)
            BH False discovery rate

        maxOutlierDeviance: float (default 0.75)
            Threshold for outlier removal (set to 0 to not remove outliers). Maximum amount
            of excess deviance carried by a single point.

        plot: bool (default True)
            Whether to display or not the mean/var relationship with selected features

        Returns
        -------
        pvals : ndarray
            Deviance p values
        deviance_outliers : boolean ndarray
            Features with a single point carrying most of excess deviance. 
        
        '''
        # In order to find potentially DE probes, find the ones that are not well modelized by the NB model using deviance as a criterion
        # For sufficiently large NB means, deviance follows a Chi-squared distribution with n samples - 1 degrees of freedom
        # But we do not have large means, so use monte-carlo instead and sample from a NB(mu, alpha) distribution
        # It is too time consuming to compute monte-carlo deviance cdfs with a sufficient sample count for each probe
        # Instead compute it at fixed percentiles of the mean and interpolate p-value estimates
        rankedMeans = (rankdata(self.means)-0.5)/len(self.means)*100
        centers = (np.linspace(0,100,evalsPts+1)+0.5/evalsPts)[:-1]
        assigned = [np.where(np.abs(rankedMeans-c) < 100.0/evalsPts)[0] for c in centers]
        pvals = np.zeros(len(rankedMeans))
        weights = np.zeros(len(rankedMeans))
        expectedDeviances = np.zeros(len(rankedMeans))
        for i in range(len(assigned)):
            bucket = assigned[i]
            interpFactor = np.maximum(0, np.abs(rankedMeans[bucket] - centers[i])*evalsPts/100)
            m = np.sum(self.means[bucket] * interpFactor)/np.sum(interpFactor)
            a = np.sum(self.regAlpha[bucket] * interpFactor)/np.sum(interpFactor)
            devBucket = self.deviances[bucket]
            ecdf = nullDeviance(m, a, self.scaled_sf)
            pvals[bucket] += np.mean(devBucket[:, None] < ecdf, axis=1) * interpFactor
            expectedDeviances[bucket] += np.mean(ecdf) * interpFactor
            weights[bucket] += interpFactor
        expectedDeviances /= weights
        pvals /= weights
        # Filter probes whose excess deviance (observed-expected) is mostly driven (75%+) by a single outlier
        deltaDev = self.deviances-expectedDeviances
        filteredOutliers = np.sum(np.square(self.anscombeResiduals.T) > maxOutlierDeviance*deltaDev[:, None], axis=1) != 1
        hv = fdrcorrection(pvals)[0] & filteredOutliers
        if plot:
            # Plot mean/variance relationship and selected probes
            v = np.var(self.normed[:, :self.normed.shape[1]], axis=0)
            m = np.mean(self.normed[:, :self.normed.shape[1]], axis=0)
            c = np.array([[0.0,0.0,1.0]]*(self.normed.shape[1]))
            c[hv] = [1.0,0.0,0.0]
            plt.figure(dpi=500)
            plt.scatter(m, v, s = 0.2, linewidths=0, c=c)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Pol II probe mean")
            plt.ylabel("Pol II probe variance")
        return pvals, filteredOutliers
        




def filterDetectableGenes(counts, readMin, expMin):
    return np.sum(counts >= readMin, axis=0) >= expMin


def quantileTransform(counts):
    rg = ((rankdata(counts, axis=0)-0.5)/counts.shape[0])*2.0 - 1.0
    return StandardScaler().fit_transform(erfinv(rg))

