
import statsmodels.api as sm
import plot_gls_res as plt
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

data = pd.read_csv(r"./Probability of Default/ALL LOANS Buckets.csv", dtype = {'curr_qtr':'str', 'CREDIT_BUCKET':'str', 'NEW_LOAN':'str', 'CURR_LOANS':'int64', 'BAD_LOANS':'int64', 'total_volume':'float'})

data["QTR2"] = pd.PeriodIndex(data["QTR"], freq='Q').to_timestamp()
data = data.sort_values(by=["QTR2"])

data["PD"] = data["BAD_LOANS"] / data["CURR_LOANS"]

data = data.loc[np.logical_and(data["CREDIT_BUCKET"] != 'No Score', data["QTR2"] > "2012-01-01"), :]

data["y"] = stats.norm.ppf(np.where(data["PD"] == 0,0.000000001, np.where(data["PD"] == 1, 1 - 0.000000001, data["PD"])))

c_bs = data["CREDIT_BUCKET"].unique()
dts = data["QTR"].unique()

names = np.array([])

i = 0

for c in c_bs:
    if c == c_bs[-1]:
        break

    data["A_" + str(c) + "_0"] = 0
    data["A_" + str(c) + "_1"] = 0

    names = np.append(names, ["A_" + str(c) + "_0", "A_" + str(c) + "_1"])

    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == '0'), ["A_" + str(c) + "_0"]] = 1
    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == '1'), ["A_" + str(c) + "_1"]] = 1


for d in dts:
    if d == dts[-1]:
        break
    data["B_" + d] = 0
    
    names = np.append(names, ["B_" + d])

    data.loc[data["QTR"] == d, ["B_" + d]] = 1

#data.to_csv(r"./Probability of Default/test.csv")

Y = data["y"].to_numpy()
X = data[names].to_numpy()
Z = data[names[np.char.find(names, "b_") == -1]]

X = sm.add_constant(X)

resid = sm.OLS(Y, X).fit().resid

VarEst = np.exp(sm.OLS(np.log(np.power(resid, 2)), Z).fit().predict())

mdl = sm.OLS(Y / VarEst, X / np.transpose(np.array([VarEst])))

res = mdl.fit()

results_as_html = res.summary().as_html()

res_out = open(r"./Model Results/GLS_SUMMARY.html", 'w')
res_out.write(results_as_html)
res_out.close()

plt.plot_gls(data, res)