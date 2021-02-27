
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

data = pd.read_excel(r"./Probability of Default/buckets.xlsx")

data = data.loc[data["CREDIT_BUCKET"] != "No Score", :]

c_bs = data["CREDIT_BUCKET"].unique()
dts = data["QTR"].unique()

names = np.array([])

i = 0

for c in c_bs:

    data["A_" + c + "_0"] = 0
    data["A_" + c + "_1"] = 0

    names = np.append(names, ["A_" + c + "_0", "A_" + c + "_1"])

    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0), ["A_" + c + "_0"]] = 1
    data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1), ["A_" + c + "_1"]] = 1


for d in dts:
    data["B_" + d] = 0
    
    names = np.append(names, ["B_" + d])

    data.loc[data["QTR"] == d, ["B_" + d]] = 1


mdl = sm.GLS(data["PD"], data[names])

res = mdl.fit()

results_as_html = res.summary().as_html()

res_out = open(r"./Model Results/GLS_SUMMARY.html", 'w')
res_out.write(results_as_html)
res_out.close()
