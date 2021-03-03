
import statsmodels.api as sm
# import plot_gls_res as plt
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

def OLS_fit():
    data = pd.read_csv(r"./Probability of Default/ALL LOANS Buckets.csv", dtype = {'curr_qtr':'str', 'CREDIT_BUCKET':'str', 'NEW_LOAN':'str', 'CURR_LOANS':'int64', 'BAD_LOANS':'int64', 'total_volume':'float'})
    
    data["QTR2"] = pd.PeriodIndex(data["QTR"].values, freq='Q').to_timestamp()
    data = data.sort_values(by=["QTR2"])
    
    data["PD"] = data["BAD_LOANS"] / data["CURR_LOANS"]
    
    data = data.loc[np.logical_and(data["CREDIT_BUCKET"] != 'No Score', data["QTR2"] > "2012-01-01"), :]
    
    data["y"] = stats.norm.ppf(np.where(data["PD"] == 0,0.000000001, np.where(data["PD"] == 1, 0.999999999, data["PD"])))
    
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
    X = data[names].to_numpy() # the dummy variable matrix with the betas
    Z = data[names[np.char.find(names, "b_") == -1]] # the dummy variable matrix without the betas
    
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)
    
    model = sm.OLS(Y, X).fit()
    
    alphas = model.params[:26]
    betas = model.params[26:]
    
    T = len(betas)
    rho_num = (1/T)*(sum(betas**2))
    rho = rho_num/(1+rho_num)
    
    gauss_PD = np.zeros(len(alphas))
    stu_t_PD = np.zeros(len(alphas))
    for i in range(len(alphas)):
        gauss_PD[i] = stats.norm.cdf(alphas[i]*np.sqrt(1-rho))
        stu_t_PD[i] = stats.t.cdf(alphas[i]*np.sqrt(1-rho), df = 4)

    return rho, gauss_PD, stu_t_PD



# res = vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), 10000)




# VarEst = np.exp(sm.OLS(np.log(np.power(resid, 2)), Z).fit().predict())

# mdl = sm.GLS(Y / VarEst, X / np.transpose(np.array([VarEst])))

# res = mdl.fit()

# results_as_html = res.summary().as_html()

# res_out = open(r"./Model Results/GLS_SUMMARY.html", 'w')
# res_out.write(results_as_html)
# res_out.close()

# plt.plot_gls(data, res)