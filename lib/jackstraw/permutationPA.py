# from : https://github.com/idc9/jackstraw
import numpy as np
from .utils import pca, svd_wrapper
from sklearn.decomposition import PCA, TruncatedSVD
def permutationPA(X, B=10, alpha=0.01, method='pca', max_rank=None):
    """
    Estimates the number of significant principal components using a permutation test.
    Adapted from https://github.com/ncchung/jackstraw and Buja and Eyuboglu (1992).

    Buja A and Eyuboglu N. (1992) Remarks on parrallel analysis. Multivariate Behavioral Research, 27(4), 509-540

    Parameters
    ----------
    X: data matrix n x d

    B (int): number of permutations

    alpha (float): cutoff value

    method (str): one of ['pca', 'svd']

    max_rank (None, int): will compute partial SVD with this rank to save
    computational time. If None, will compute full SVD.
    """

    if method == 'pca':
        decomp = PCA
    elif method == 'svd':
        decomp = TruncatedSVD
    else:
        raise ValueError('{} is invalid method'.format(method))

    # compute eigenvalues of observed data
    ref = decomp(max_rank)
    ref.fit(X)
    # squared frobinius norm of the matrix, also equal to the
    # sum of squared eigenvalues
    dstat_obs = ref.explained_variance_

    # compute premutation eigenvalues
    dstat_null = np.zeros((B, len(dstat_obs)))
    np.random.seed(42)
    for b in range(B):
        X_perm = np.apply_along_axis(np.random.permutation, 0, X)
        perm = decomp(max_rank)
        perm.fit(X_perm)
        dstat_null[b, :] = perm.explained_variance_

    # compute p values
    pvals = np.ones(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = np.mean(dstat_null[:, i] >= dstat_obs[i])
    for i in range(1, len(dstat_obs)):
        pvals[i] = max(pvals[i - 1], pvals[i])

    # estimate rank
    r_est = sum(pvals <= alpha)

    return r_est, pvals


def permutationPA_PCA(X, B=3, alpha=0.01, method='pca', max_rank=None, mincomp=0):
    """
    Estimates the number of significant principal components using a permutation test.
    Adapted from https://github.com/ncchung/jackstraw and Buja and Eyuboglu (1992).

    Buja A and Eyuboglu N. (1992) Remarks on parrallel analysis. Multivariate Behavioral Research, 27(4), 509-540

    Parameters
    ----------
    X: data matrix n x d

    B (int): number of permutations

    alpha (float): cutoff value

    method (str): one of ['pca', 'svd']

    max_rank (None, int): will compute partial SVD with this rank to save
    computational time. If None, will compute full SVD.
    """

    if method == 'pca':
        decomp = PCA
    elif method == 'svd':
        decomp = TruncatedSVD
    else:
        raise ValueError('{} is invalid method'.format(method))

    # Compute eigenvalues of observed data
    ref = decomp(max_rank, whiten=True, svd_solver="arpack", random_state=42)
    decompRef = ref.fit_transform(X)
    dstat_obs = ref.explained_variance_
    # Compute permutation eigenvalues
    dstat_null = np.zeros((B, len(dstat_obs)))
    np.random.seed(42)
    for b in range(B):
        X_perm = np.apply_along_axis(np.random.permutation, 0, X)
        perm = decomp(max_rank, svd_solver="arpack", random_state=42)
        perm.fit(X_perm)
        dstat_null[b, :] = perm.explained_variance_

    # compute p values
    pvals = np.ones(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = np.mean(dstat_null[:, i] >= dstat_obs[i])
    for i in range(1, len(dstat_obs)):
        pvals[i] = 1.0-(1.0-pvals[i - 1])*(1.0-pvals[i])

    # estimate rank
    r_est = max(sum(pvals <= alpha),mincomp)

    return decompRef[:, :r_est]