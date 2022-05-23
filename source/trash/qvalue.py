import numpy as np
import numba 

@numba.njit()
def qvalue(pvals):
    order = np.argsort(pvals)
    invOrder = np.argsort(order)
    pvals = pvals[order]
    # Estimate proportion of null hypotheses
    pos = np.linspace(0.01, 0.99, 100)
    pik = [np.mean(pvals > p) / (1-p) for p in pos]
    pi0 = np.percentile(pik, 5)
    # Compute q-values
    qvals = np.zeros_like(pvals)
    qvals[-1] = pi0*pvals[-1]
    for i in np.arange(len(pvals)-2, -1, -1):
        qvals[i] = min(pi0*len(pvals)*pvals[i]/(i+1.0), qvals[i+1])
    return qvals[invOrder]
