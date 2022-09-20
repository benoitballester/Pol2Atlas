# %%
import numpy as np
import numba as nb
from scipy.stats import ttest_ind
import time
import os
from joblib import delayed, Parallel


maxT = len(os.sched_getaffinity(0))

@nb.njit(nb.float64(nb.float32[:], nb.float32), fastmath=True)
def unbiasedStd(x, m):
    return np.sqrt(np.sum(np.square(x-m))/(len(x)-1))

@nb.njit(nb.float64(nb.float32[:], nb.float32[:]), fastmath=True)
def computeWelshT(x, y):
    m1 = np.mean(x)
    m2 = np.mean(y)
    v1 = unbiasedStd(x, m1)
    v2 = unbiasedStd(y, m2)
    sx1 = v1 / np.sqrt(len(x))
    sx2 = v2 / np.sqrt(len(y))
    return (m1 - m2) / np.sqrt(sx1*sx1 + sx2*sx2)

@nb.njit(nb.float64[:](nb.float32[:,:], nb.float32[:,:]), fastmath=True)
def columnWiseT(x, y):
    tstats = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        tstats[i] = computeWelshT(x[:, i], y[:, i])
    return tstats

@nb.njit(nb.float64[:](nb.float32[:,:], nb.boolean[:], nb.int64), fastmath=True)
def permutation_welsh_T_greater(x, group, perms):
    obs = columnWiseT(x[group], x[np.logical_not(group)])
    sups = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        col = x[:, i]
        for j in range(perms):
            permGrp = np.random.permutation(group)
            sups[i] += obs[i] < computeWelshT(col[permGrp], col[np.logical_not(permGrp)])
    return (sups + 1) / (perms+1)

@nb.njit(nb.float64[:](nb.float32[:,:], nb.boolean[:], nb.int64), fastmath=True)
def permutation_welsh_T_less(x, group, perms):
    obs = columnWiseT(x[group], x[np.logical_not(group)])
    sups = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        col = x[:, i]
        for j in range(perms):
            permGrp = np.random.permutation(group)
            sups[i] += obs[i] > computeWelshT(col[permGrp], col[np.logical_not(permGrp)])
    return (sups + 1) / (perms+1)

@nb.njit(nb.float64[:](nb.float32[:,:], nb.boolean[:], nb.int64), fastmath=True)
def permutation_welsh_T_twosided(x, group, perms):
    obs = np.abs(columnWiseT(x[group], x[np.logical_not(group)]))
    sups = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        col = x[:, i]
        for j in range(perms):
            permGrp = np.random.permutation(group)
            sups[i] += obs[i] > np.abs(computeWelshT(col[permGrp], col[np.logical_not(permGrp)]))
    return (sups + 1) / (perms+1)

def welshTperm(x, groups, perms=100, alternative="two-sided", workerPool=None):
    if alternative == "two-sided":
        func = permutation_welsh_T_twosided
    if alternative == "greater":
        func = permutation_welsh_T_greater
    if alternative == "less":
        func = permutation_welsh_T_less
    if workerPool == None:
        pvals = func(x, groups, perms)
        return pvals
    else:
        blockSize = int(x.shape[1]/len(os.sched_getaffinity(0))/2+1)
        blockPos = np.append(np.arange(0, x.shape[1], blockSize), x.shape[1])
        blocks = np.split(x, blockPos, 1)
        pvals = workerPool(delayed(func)(block, groups, perms) for block in blocks)
        return np.concatenate(pvals)


# %%
if __name__ == "__main__":
    r1 = np.random.normal(loc=0.0, size=(490,180000)).astype("float32")
    r2 = np.random.normal(size=(20,180000)).astype("float32")
    r = np.copy(np.vstack([r1, r2]))
    grps = np.array([True]*490 + [False]*20)
    t = time.time()
    with Parallel(n_jobs=-1, verbose=1, batch_size=1, max_nbytes=None) as pool:
        pvals = welshTperm(r, grps, perms=100, alternative="less", workerPool=None)
    print(time.time()-t)    
    
    """ t = time.time()
    pvals = ttest_ind(r1, r2, equal_var=False, axis=0, permutations=100, alternative="greater")[1]
    print(time.time()-t)  """
# %%
