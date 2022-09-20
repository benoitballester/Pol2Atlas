# %%
# Everything here is mostly adapted from scipy
import numpy as np
from scipy.stats import norm, rankdata, mannwhitneyu
from collections import namedtuple

def _tie_term(ranks):
    """Tie correction term"""
    # element i of t is the number of elements sharing rank i
    _, t = np.unique(ranks, return_counts=True, axis=-1)
    return (t**3 - t).sum(axis=-1)

def _get_mwu_z(U, n1, n2, ranks, axis=0, continuity=True, tie_term=None):
    '''Standardized MWU statistic'''
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2]
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

    # equivalent to using scipy.stats.tiecorrect
    # T = np.apply_along_axis(stats.tiecorrect, -1, ranks)
    # s = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)

    numerator = U - mu

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    if continuity:
        numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z

MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

class mannWhitneyAsymp:
    '''
    Mann-whitney U test with reusable rank calculations
    Only the asymptotic normal approximation is implemented
    '''
    def __init__(self, values):
        self.ranks = rankdata(values, axis=0)
        self.tie_term = np.apply_along_axis(_tie_term, 0, self.ranks)
    
    def test(self, group, alternative="two-sided"):
        '''
        group : array-like of len(values)
            Array of only 0s or 1s.
        '''
        group = np.array(group)
        n1 = np.sum(group == 0)
        n2 = np.sum(group == 1)
        R1 = self.ranks[group == 0].sum(axis=0)
        U1 = R1 - n1*(n1+1)/2
        U2 = n1*n2 - U1
        if alternative == "greater":
            U, f = U1, 1  # U is the statistic to use for p-value, f is a factor
        elif alternative == "less":
            U, f = U2, 1  # Due to symmetry, use SF of U2 rather than CDF of U1
        else:
            U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test
        print(U.shape)
        z = _get_mwu_z(U, n1, n2, self.ranks, continuity=True, tie_term=self.tie_term)
        p = norm.sf(z)
        p *= f
        # Ensure that test statistic is not greater than 1
        # This could happen for exact test when U = m*n/2
        p = np.clip(p, 0, 1)
        return MannwhitneyuResult(U1, p)
        

if __name__ == "__main__":
    arr1 = np.random.randint(5,15,(100,2))
    arr2 = np.random.normal(70,205,(100,2))
    concat = np.concatenate([arr1, arr2], axis=0)
    labels = [0]*100 + [1]*100
    tester = mannWhitneyAsymp(concat)
    print(tester.test(labels, alternative="greater"))
    print(mannwhitneyu(arr1, arr2, method="asymptotic", alternative="greater"))