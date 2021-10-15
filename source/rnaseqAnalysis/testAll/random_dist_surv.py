# %%
import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from lifelines.statistics import logrank_test
from rpy2.robjects.packages import importr
pandas2ri.activate()
maxstat = importr("maxstat")
survival = importr("survival")
survminer = importr("survminer")

n = 1000
allPvalsMaxRank = []
allPvalsLogRank = []
for i in range(n):
    # Random values, time to event, and events
    df = pd.DataFrame()
    df["Val"] = np.random.normal(0.0,1.0,100)
    df["TTE"] = np.random.randint(1,1000,100)
    df["Event"] = np.random.randint(0,2,100).astype(bool)
    r_dataframe = ro.conversion.py2rpy(df)
    fml = ro.r(f"Surv(TTE, Event) ~ Val")
    # Compute cutoff point and max rank p-value
    mstat = maxstat.maxstat_test(fml, data=r_dataframe, smethod="LogRank", pmethod="Lau94")
    pval = mstat.rx2('p.value')[0]
    allPvalsMaxRank.append(pval)
    # P-value after cutoff selection
    gr1 = df["Val"] < mstat.rx2("estimate")[0]
    gr2 = np.logical_not(gr1)
    allPvalsLogRank.append(logrank_test(df["TTE"][gr1], df["TTE"][gr2], df["Event"][gr1], df["Event"][gr2]).p_value)
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=300)
plt.title("Distribution of Monte carlo max rank p-values on a random dataset")
corr = allPvalsMaxRank/np.max(allPvalsMaxRank)
plt.hist(corr, 20, density=True)
plt.show()
plt.figure()
plt.figure(dpi=300)
plt.title("Distribution of p-values computed after cutoff selection on a random dataset")
plt.hist(allPvalsLogRank, 20, density=True)
plt.show()
# %%
