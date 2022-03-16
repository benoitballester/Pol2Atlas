# %%
import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.discrete import discrete_model
from statsmodels.genmod.families import NegativeBinomial
import matplotlib.pyplot as plt
nbMean = 1.0
alpha = 5.0
nSamples = 10

n = 1 / alpha
p = nbMean / (nbMean + alpha * (nbMean**2))
estAlpha = []
estAlpha2 = []
for i in range(100):
    try:
        s1 = ss.nbinom.rvs(n,p,size=10)
        reg = discrete_model.NegativeBinomial(s1, np.ones_like(s1)).fit(disp=0, method="nm", ftol=1e-9, maxiter=500)
        estAlpha.append(reg.params[1])
        estAlpha2.append((s1.var() - s1.mean())/s1.mean()**2)
    except:
        continue
estAlpha = np.array(estAlpha)
print(np.mean(estAlpha))
print(np.median(estAlpha))
print(ss.mstats.hmean(estAlpha))
plt.hist(estAlpha)
plt.show()
plt.hist(s1,40)
# %%
