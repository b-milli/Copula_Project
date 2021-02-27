
import Functions.datafunctions as df
import Functions.vasicek_loop as vl
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


delinq_data = df.get_data()

last_pd = {}
score_b = delinq_data["CREDIT_BUCKET"].unique()[delinq_data["CREDIT_BUCKET"].unique() != "No Score"]
for c in score_b:
    delinq_vect_old = (delinq_data.loc[np.logical_and(delinq_data["CREDIT_BUCKET"] == c, np.logical_and(delinq_data["new_loan"] == 0, delinq_data["cur_qtr"] == "2020-Q1")),["DELINQ_IND"]]).to_numpy()
    delinq_vect_new = (delinq_data.loc[np.logical_and(delinq_data["CREDIT_BUCKET"] == c, np.logical_and(delinq_data["new_loan"] == 1, delinq_data["cur_qtr"] == "2020-Q1")),["DELINQ_IND"]]).to_numpy()

    #print(c,np.sum(delinq_vect_new),len(delinq_vect_new))
    try:
        last_pd[c] = {"Old Loan":np.sum(delinq_vect_new) / len(delinq_vect_new), "New Loan":np.sum(delinq_vect_old) / len(delinq_vect_old)}
    except:
        print(c)


curr = delinq_data.loc[np.logical_and(delinq_data["cur_qtr"] == "2020-Q1", delinq_data["CREDIT_BUCKET"] != "No Score"),:]

curr = curr.reset_index()

tot_volume = np.sum(curr["CURR_UPB"].to_numpy())

loan_weight = np.zeros(len(curr.index))
loan_corr = np.zeros(len(curr.index)) + 0.15
loan_pd = np.zeros(len(curr.index))

for index, row in curr.iterrows():
    loan_weight[index] = row["CURR_UPB"] / tot_volume

    c_bucket = row["CREDIT_BUCKET"]

    new_ind = "New Loan"
    if row["new_loan"] == 0:
        new_ind = "Old Loan"

    loan_pd[index] = last_pd[c_bucket][new_ind]

res = vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), 100)

plt.figure(figsize=(20,15))

plt.hist(res, bins = 5000, density = True)
plt.title("Gaussian Distribution of Losses")
textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * 0.01))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[0:int(np.floor(len(res) * 0.01))])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Gaussian_Distribution.png", dpi = 600)
plt.cla()

res = vl.Copula_Loop(loan_weight, loan_corr, loan_pd, lambda x: stats.t.rvs(size = x, df = 2), lambda x: stats.t.ppf(x, df = 2), lambda x: stats.t.cdf(x, df = 2), 100)

plt.figure(figsize=(20,15))

plt.hist(res, bins = 5000, density = True)
plt.title("Student Distribution of Losses")
textstr =  r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
            "\n" + \
             r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * 0.01))] * 100,3)) + \
             "\n Expected Shortfall = {}".format(np.round(np.mean(res[0:int(np.floor(len(res) * 0.01))])*100, 3)) + \
             "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
#Style
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Append the Var and ES
plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
#Save and Clear
plt.savefig(r"Plots\Student_Distribution.png", dpi = 600)