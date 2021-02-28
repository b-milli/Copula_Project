
import statsmodels.api as sm
import plot_gls_res as plt
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

data = pd.read_excel(r"./Probability of Default/buckets.xlsx")

data["QTR2"] = pd.PeriodIndex(data["QTR"], freq='Q').to_timestamp()
data = data.sort_values(by=["QTR2"])

data = data.loc[np.logical_and(np.logical_and(data["CREDIT_BUCKET"] != "No Score", data["QTR2"] > "2012-01-01"), data["QTR2"] < "2020-01-01"), :]

data["y"] = stats.norm.ppf(np.where(data["PD"] == 0,0.000000001, np.where(data["PD"] == 1, 0.999999999, data["PD"])))

c_bs = data["CREDIT_BUCKET"].unique()
dts = data["QTR"].unique()

names = np.array([])

i = 0

for c in c_bs:
    if c == c_bs[-1]:
        break

    data["A_" + c + "_0"] = 0
    data["A_" + c + "_1"] = 0

    names = np.append(names, ["A_" + c + "_0", "A_" + c + "_1"])

    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0), ["A_" + c + "_0"]] = 1
    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1), ["A_" + c + "_1"]] = 1


for d in dts:
    if d == dts[-1]:
        break
    data["B_" + d] = 0
    
    names = np.append(names, ["B_" + d])

    data.loc[data["QTR"] == d, ["B_" + d]] = 1


# ols_resid = sm.OLS(data["y"], data[names]).fit().resid

# var = np.zeros(len(c_bs) * 2)
# i=0
# for c in c_bs:
#     tmp_resid = ols_resid[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0)]
#     var[i] = np.dot(tmp_resid, np.transpose(tmp_resid)) / (len(tmp_resid) - 2)
#     i = i + 1
#     tmp_resid = ols_resid[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1)]
#     var[i] = np.dot(tmp_resid, np.transpose(tmp_resid)) / (len(tmp_resid) - 2)
#     i = i + 1

# X = data.loc[:,names].to_numpy()
# H = np.dot(np.dot(X,np.linalg.inv(np.dot(np.transpose(X), X))),np.transpose(X))
# cov = np.diag(var) - np.dot(np.dot(X,np.linalg.inv(np.dot(np.transpose(X), X))),np.transpose(X))


mdl = sm.GLS(data["y"], data[names])

res = mdl.fit()

results_as_html = res.summary().as_html()

res_out = open(r"./Model Results/GLS_SUMMARY.html", 'w')
res_out.write(results_as_html)
res_out.close()

# resid0 = res.resid
# pred0 = res.predict()

# def mle_fit(params, data,names):

#     log_liklehood = 0

#     for index, row in data.iterrows():
#             log_liklehood += np.log(stats.norm.pdf(row['y'] - np.dot(params, row[names].to_numpy())))

#     return log_liklehood

# opt_res = minimize(lambda x: mle_fit(x, data, names), np.ones(len(names)), method='Nelder-Mead')

# opt_res_df = pd.DataFrame({"param":names, "fit":opt_res.x})

# opt_res_df.to_csv("./Model Results/MLE_norm_Optimization.csv")

plt.plot_gls(data, res)