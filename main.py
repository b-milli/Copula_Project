
import Functions.datafunctions as df
import Functions.vasicek_loop as vl
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


all_data = df.get_data()

curr = all_data.loc[all_data["QTR"] == "2020-Q1",:]

tot_volume = np.sum(curr["total_volume"].to_numpy())

curr = curr.reset_index()

loan_weight = np.zeros(len(curr.index))
loan_corr = np.zeros(len(curr.index)) + 0.15
loan_pd = np.zeros(len(curr.index))

for index, row in curr.iterrows():
    loan_weight[index] = row["total_volume"] / tot_volume

    loan_pd[index] = np.mean(curr.loc[np.logical_and(all_data["CREDIT_BUCKET"] == row["CREDIT_BUCKET"], all_data["NEW_LOAN"] == row["NEW_LOAN"]), "PD"])

res = np.sort(vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), 100000))

plt.figure(figsize=(10,7.5))

plt.hist(res, bins = 1000, density = True)
plt.title("Gaussian Distribution of Losses - Base Case")
textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * 0.99))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[int(np.floor(len(res) * 0.99)):len(res)])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Gaussian_Distribution_Base.png", dpi = 600)
plt.cla()

res = np.sort(vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.t.rvs(size = x, df = 2), lambda x: stats.t.ppf(x, df = 2), lambda x: stats.t.cdf(x, df = 2), 100000))

plt.figure(figsize=(10,7.5))

plt.hist(res, bins = 1000, density = True)
plt.title("Student Distribution of Losses - Base Case")
textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * (1 - 0.01)))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[int(np.floor(len(res) * (1 - 0.01))):len(res)])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Student_Distribution_Base.png", dpi = 600)


fit_data = pd.read_csv("Probability of Default/fit_model.csv")

loan_weight = np.zeros(len(curr.index))
loan_corr = np.zeros(len(curr.index)) + np.max(fit_data["norm_rho"].to_numpy())
loan_pd = np.zeros(len(curr.index))

for index, row in curr.iterrows():
    loan_weight[index] = row["total_volume"] / tot_volume

    loan_pd[index] = fit_data.loc[np.logical_and(fit_data["CREDIT_BUCKET"] == row["CREDIT_BUCKET"], fit_data["NEW_LOAN"] == row["NEW_LOAN"]),"norm_PD"]

res = np.sort(vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), 100000))

plt.figure(figsize=(10,7.5))

plt.hist(res, bins = 1000, density = True)
plt.title("Gaussian Distribution of Losses - Fit Parameters")
textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * (1 - 0.01)))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[int(np.floor(len(res) * (1 - 0.01))):len(res)])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Gaussian_Distribution_Fit.png", dpi = 600)
plt.cla()

loan_weight = np.zeros(len(curr.index))
loan_corr = np.zeros(len(curr.index)) + np.max(fit_data["student_rho"].to_numpy())
loan_pd = np.zeros(len(curr.index))

for index, row in curr.iterrows():
    loan_weight[index] = row["total_volume"] / tot_volume

    loan_pd[index] = fit_data.loc[np.logical_and(fit_data["CREDIT_BUCKET"] == row["CREDIT_BUCKET"], fit_data["NEW_LOAN"] == row["NEW_LOAN"]),"student_PD"]

res = np.sort(vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.t.rvs(size = x, df = 2), lambda x: stats.t.ppf(x, df = 2), lambda x: stats.t.cdf(x, df = 2), 100000))

plt.figure(figsize=(10,7.5))

plt.hist(res, bins = 1000, density = True)
plt.title("Student Distribution of Losses - Fit Parameters")
textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * (1 - 0.01)))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[int(np.floor(len(res) * (1 - 0.01))):len(res)])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Student_Distribution_Fit.png", dpi = 600)