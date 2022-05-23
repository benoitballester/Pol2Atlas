import warnings

import KDEpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import scipy.stats as ss
import statsmodels.discrete.discrete_model as discrete_model
from joblib import Parallel, delayed
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.special import erfinv
from scipy.stats import nbinom, rankdata, norm
from sklearn.decomposition import PCA
from statsmodels.api import GLM
from statsmodels.genmod.families import family
from statsmodels.stats.multitest import fdrcorrection

scran = importr("scran")

def findMode(arr):
    # Finds the modal value of a continuous sample
    pos, fitted = KDEpy.FFTKDE(bw="silverman").fit(arr).evaluate(1000000)
    return pos[np.argmax(fitted)]

def nb_rqr(x, m, alpha):
    n = 1/alpha
    p = m / (m + alpha * (m**2))
    q = nbinom(n,p).cdf(x-1) + np.random.random(x.shape) * nbinom(n,p).pmf(x)
    f1 = norm.ppf(q)
    f2 = norm.sf(q)
    return np.clip(np.where(q > 0, f1, f2), -38.2, 38.2)

def statsProcess(regAlpha,scaled_ni,countsSel,means):
    func = family.NegativeBinomial(alpha=regAlpha)
    pred = means * scaled_ni
    anscombeResiduals = func.resid_dev(countsSel, pred)
    dev_raw = func.deviance(countsSel, pred)
    res_pearson = (countsSel-pred)/np.sqrt(pred + regAlpha * np.square(pred))
    return anscombeResiduals, dev_raw, res_pearson, pred

def devianceP(nbMean, alpha, devRef, nSamples = 100):
    alpha = alpha
    n = 1 / alpha
    p = nbMean / (nbMean + alpha * (nbMean**2))
    devs = np.zeros(nSamples)
    rv = np.random.negative_binomial(n,p,size=(nSamples,len(p)))
    func = family.NegativeBinomial(alpha=alpha)
    resid = func.resid_dev(rv, nbMean) + 0.25
    devs = np.sum(np.square(resid), axis=1)
    return np.mean(devs > devRef), np.mean(devs)

def nullDeviance(nbMean, alpha, nSamples = 10000):
    alpha = alpha
    n = 1 / alpha
    p = nbMean / (nbMean + alpha * (nbMean**2))
    devs = np.zeros(nSamples)
    rv = np.random.negative_binomial(n,p,size=(nSamples,len(p)))
    func = family.NegativeBinomial(alpha=alpha)
    resid = func.resid_dev(rv, nbMean) + 0.25
    devs = np.sum(np.square(resid), axis=1)
    return devs

def fitModels(counts, sfs, m):
    warnings.filterwarnings("error")
    try:
        model = discrete_model.NegativeBinomial(counts, np.ones(len(counts)), exposure=sfs)
        fit = model.fit([np.log(m), 10.0], method="nm", ftol=1e-9, maxiter=500, disp=False, skip_hessian=True)
    except:
        return [np.log(m), -1]
    warnings.filterwarnings("default")
    return fit.params


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
        np.random.seed(42)
        subSampleEst = np.minimum(subSampleEst, counts.shape[1])
        if subSampleEst == -1:
            subSampleEst = counts.shape[1]
        shuffled = np.random.permutation(counts.shape[1])[:subSampleEst]
        m = np.mean(self.normed, axis=0)
        with Parallel(n_jobs=-1, verbose=verbose, batch_size=512) as pool:
           fittedParams = pool(delayed(fitModels)(counts[:, i], self.scaled_sf, m[i]) for i in shuffled)
        alphas = np.array(fittedParams)[:, 1]
        valid = alphas > 1e-3
        # Estimate NB overdispersion in function of mean expression
        # Overdispersion can be caused by biological variation as well as technical variation
        # To estimate technical overdispersion, it is assumed that a large fraction of probes are non-DE, and that overdispersion is a function of mean
        # The modal value of overdispersion is tracked for each decile of mean expression using a kernel density estimate
        # The median value would overestimate overdispersion as the % of DE probes is unknown
        means = np.mean(self.normed[:, shuffled], axis=0)
        nQuantiles = 10
        pcts = np.linspace(0,100,nQuantiles+1)
        centers = (pcts * 0.5 + np.roll(pcts,1)*0.5)[1:]/100
        quantiles = np.percentile(means, pcts)
        quantiles[0] = 0
        digitized = np.digitize(means, quantiles)
        regressed = []
        for i in np.arange(1,nQuantiles+1):
            regressed.append(findMode(alphas[valid & (digitized == i)]))
        regressed = np.array(regressed).ravel()
        self.means = np.mean(self.normed, axis=0)
        fittedAlpha = si.interp1d(centers, regressed, bounds_error=False, fill_value="extrapolate")
        self.regAlpha = fittedAlpha((rankdata(self.means)-0.5)/len(self.means))
        if plot:
            plt.figure(dpi=500)
            plt.plot(self.regAlpha[np.argsort(np.mean(self.normed, axis=0))])
            plt.scatter(np.argsort(np.argsort(np.mean(self.normed[:, shuffled], axis=0)))*self.normed.shape[1]/len(alphas), alphas, s=0.5, linewidths=0, c="red")
            # plt.yscale("log")
            plt.xlabel("Pol II ranked mean expression")
            plt.ylim(1e-2, 1e2)
            plt.ylabel("Alpha (overdispersion)")
            plt.show()
            plt.close()
            # Plot mean/variance relationship and selected probes
            v = np.var(self.normed[:, :self.normed.shape[1]], axis=0)
            m = np.mean(self.normed[:, :self.normed.shape[1]], axis=0)
            c = np.array([[0.0,0.0,1.0]]*(self.normed.shape[1]))
            plt.figure(dpi=500)
            plt.scatter(m, v, s = 0.2, linewidths=0, c=c)
            plt.scatter(m, m+m*m*self.regAlpha, s = 1.0, linewidths=0, c=[0.0,1.0,0.0])
            pts = np.geomspace(m.min(), m.max())
            plt.plot(pts, pts)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Pol II probe mean")
            plt.ylabel("Pol II probe variance")
            plt.show()
            plt.close()
        # Dispatch accross multiple processes
        with Parallel(n_jobs=-1, verbose=verbose, batch_size=512) as pool:
            stats = pool(delayed(statsProcess)(self.regAlpha[i], self.scaled_sf, counts[:, i], self.means[i]) for i in range(counts.shape[1]))
        # stats = [statsProcess(self.regAlpha[i], self.scaled_sf, counts[:, i], self.means[i]) for i in range(counts.shape[1])]
        # Unpack results
        print("lol")
        self.anscombeResiduals = np.array([i[0] for i in stats]).T
        self.anscombeResiduals = np.nan_to_num(self.anscombeResiduals, nan=0.0)
        self.deviances = np.sum(self.anscombeResiduals**2, axis=0)
        self.res_dev = np.array([i[2] for i in stats]).T
        self.nullDataset = np.array([i[3] for i in stats]).T
        return self

    
    def hv_selection(self, evalsPts=100, maxOutlierDeviance=0.66, plot=True, verbose=True):
        '''
        Select features not not well modelized by the fitted NB model.
        Requires to have run the fit method before.

        Parameters
        ----------
        evalsPts: int (default 100)
            The number of points at which to evaluate the CDF of the deviance

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
        with Parallel(n_jobs=-1, verbose=verbose, batch_size=512) as pool:
            stats = pool(delayed(devianceP)(self.nullDataset[:, i], self.regAlpha[i], self.deviances[i]) for i in range(self.normed.shape[1]))
        pvals = np.array([i[0] for i in stats])
        expectedDeviances = np.array([i[1] for i in stats])

        # It is too time consuming to compute monte-carlo deviance cdfs with a sufficient sample count for each probe
        # Instead compute it at fixed percentiles of the mean and interpolate p-value estimates
        '''
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
            ecdf = nullDeviance(m * self.scaled_sf, a)
            pvals[bucket] += np.mean(devBucket[:, None] < ecdf, axis=1) * interpFactor
            expectedDeviances[bucket] += np.mean(ecdf) * interpFactor
            weights[bucket] += interpFactor
        expectedDeviances /= weights
        pvals /= weights
        '''
        # Filter probes who
        # Filter probes whose excess deviance (observed-expected) is mostly driven (75%+) by a single outlier
        deltaDev = self.deviances-expectedDeviances
        filteredOutliers = np.sum(np.square(self.anscombeResiduals.T) > maxOutlierDeviance*deltaDev[:, None], axis=1) != 1
        hv = fdrcorrection(pvals)[0]
        if plot:
            # Plot mean/variance relationship and selected probes
            v = np.var(self.normed[:, :self.normed.shape[1]], axis=0)
            m = np.mean(self.normed[:, :self.normed.shape[1]], axis=0)
            c = np.array([[0.0,0.0,1.0]]*(self.normed.shape[1]))
            c[hv] = [1.0,0.0,0.0]
            plt.figure(dpi=500)
            plt.scatter(m, v, s = 0.2, linewidths=0, c=c)
            plt.scatter(m, m+m*m*self.regAlpha, s = 1.0, linewidths=0, c=[0.0,1.0,0.0])
            pts = np.geomspace(m.min(), m.max())
            plt.plot(pts, pts)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Pol II probe mean")
            plt.ylabel("Pol II probe variance")
            plt.show()
        return pvals, filteredOutliers
        

def permutationPA_PCA(X, perm=3, alpha=0.01, solver="randomized", whiten=True,
                      max_rank=None, mincomp=0, returnModel=False, plot=True):
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
        perm = PCA(max_rank, svd_solver=solver, random_state=42)
        perm.fit(X_perm)
        dstat_null[b, :] = perm.explained_variance_

    # Compute p values
    pvals = np.ones(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = np.mean(dstat_null[:, i] >= dstat_obs[i])
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


def filterDetectableGenes(counts, readMin, expMin):
    return np.sum(counts >= readMin, axis=0) >= expMin


def quantileTransform(counts):
    rg = ((rankdata(counts, axis=0)-0.5)/counts.shape[0])*2.0 - 1.0
    return StandardScaler().fit_transform(erfinv(rg))

