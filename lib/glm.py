import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
from statsmodels.genmod.families.family import NegativeBinomial


def waldTestNB(counts, labels, alphas, exposure):
    warnings.filterwarnings("error")
    pvals = []
    for i in range(counts.shape[1]):
        df = pd.DataFrame(np.array([labels.astype(float), np.ones_like(labels)]).T, 
                                columns=["GS", "Intercept"])
        try:
            nb_reg = sm.GLM(pd.DataFrame(counts[:, i]), df, family=NegativeBinomial(alpha=alphas[i]), exposure=exposure).fit(disp=0)
            pvals.append(nb_reg.pvalues["GS"])
        except:
            pvals.append(1.0)
        if i % 1000 == 0:
            print(i)
    warnings.filterwarnings("ignore")
    return pvals