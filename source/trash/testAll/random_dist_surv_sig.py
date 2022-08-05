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
    df1 = pd.DataFrame()
    df1["Val"] = np.random.normal(0.0,1.0,100)
    df1["TTE"] = np.random.randint(1,1000,100)
    df1["Event"] = np.random.randint(0,2,100).astype(bool)
    df2 = pd.DataFrame()
    df2["Val"] = np.random.normal(3.0,1.0,100)
    df2["TTE"] = np.random.randint(1,1000,100)
    df2["Event"] = np.random.random(100) > 0.75
    df = pd.concat([df1, df2])
    r_dataframe = ro.conversion.py2rpy(df)
    fml = ro.r(f"Surv(TTE, Event) ~ Val")
    # Compute cutoff point and max rank p-value
    mstat = maxstat.maxstat_test(fml, data=r_dataframe, smethod="LogRank", pmethod="Lau92")
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
plt.hist(allPvalsMaxRank, 20, density=True)
plt.show()
plt.figure()
plt.figure(dpi=300)
plt.title("Distribution of p-values computed after cutoff selection on a random dataset")
plt.hist(allPvalsLogRank, 20, density=True)
plt.show()
# %%
