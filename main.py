
import Functions.datafunctions as df
import Functions.vasicek_loop as vl
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing



def gaus_base(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight,base_loan_corr, base_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), trials))

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
    plt.savefig(r"Model Results\Plots\Gaussian_Distribution_Base.png", dpi = 600)
    plt.cla()
    

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["Gaussian_Base_VAR"] = np.nan
    var_dists["Gaussian_Base_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"Gaussian_Base_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"Gaussian_Base_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["gauss_base"] = var_dists


def st_base_3(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, base_loan_corr, base_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 3), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 3$")
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
    plt.savefig(r"Model Results\Plots\Student_Distribution_base_3.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Base_3_VAR"] = np.nan
    var_dists["ST_Base_3_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Base_3_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Base_3_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_base_3"] = var_dists

def st_base_10(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, base_loan_corr, base_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 10), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 10$")
    textstr = r"Mean = {}".format(np.round(np.mean(res)*100,3)) + \
                "\n" + \
                r'$99\%$ Var = {}'.format(np.round(res[int(np.floor(len(res) * 0.99))] * 100,3)) + \
                "\n Expected Shortfall = {}".format(np.round(np.mean(res[int(np.floor(len(res) * 0.99)):len(res)]), 3)) + \
                "\n Standard Error = {}".format(np.round(stats.sem(res, axis = None), 5))
    #Style
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #Append the Var and ES
    plt.text((plt.xlim()[1] - plt.xlim()[0]) * 0.05 + plt.xlim()[0], plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])*0.05, textstr, fontsize=14,verticalalignment='top', bbox=props)
    #Save and Clear
    plt.savefig(r"Model Results\Plots\Student_Distribution_base_10.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Base_10_VAR"] = np.nan
    var_dists["ST_Base_10_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Base_10_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Base_10_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_base_10"] = var_dists

def st_base_30(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, base_loan_corr, base_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 30), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 30$")
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
    plt.savefig(r"Model Results\Plots\Student_Distribution_base_30.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Base_30_VAR"] = np.nan
    var_dists["ST_Base_30_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Base_30_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Base_30_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_base_30"] = var_dists




def gaus_fit(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, fit_loan_corr, fit_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: stats.norm.ppf(x), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Gaussian Distribution of Losses - Fit Parameters")
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
    plt.savefig(r"Model Results\Plots\Gaussian_Distribution_Fit.png", dpi = 600)
    plt.cla()

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["Gaussian_Fit_VAR"] = np.nan
    var_dists["Gaussian_Fit_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"Gaussian_Fit_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"Gaussian_Fit_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["gauss_fit"] = var_dists




def st_fit_3(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, fit_loan_corr, fit_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 3), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 3$")
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
    plt.savefig(r"Model Results\Plots\Student_Distribution_Fit_3.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Fit_3_VAR"] = np.nan
    var_dists["ST_Fit_3_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Fit_3_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Fit_3_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_fit_3"] = var_dists



def st_fit_10(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, fit_loan_corr, fit_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 10), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 10$")
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
    plt.savefig(r"Model Results\Plots\Student_Distribution_Fit_10.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Fit_10_VAR"] = np.nan
    var_dists["ST_Fit_10_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Fit_10_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Fit_10_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_fit_10"] = var_dists


def st_fit_30(return_dict):
    res = np.sort(vl.Copula_Loop(loan_weight, fit_loan_corr, fit_loan_pd, lambda x: stats.norm.rvs(size = x), lambda x: t_inv(x, df = 30), lambda x: stats.norm.cdf(x), trials))

    plt.figure(figsize=(10,7.5))

    plt.hist(res, bins = 1000, density = True)
    plt.title("Student Distribution of Losses - Fit Parameters \n" + r"$\nu = 30$")
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
    plt.savefig(r"Model Results\Plots\Student_Distribution_Fit_30.png", dpi = 600)

    var_dists = pd.DataFrame({"var_perc" : np.linspace(.90, .99, 1000)})
    var_dists["ST_Fit_30_VAR"] = np.nan
    var_dists["ST_Fit_30_ES"] = np.nan

    for index, row in var_dists.iterrows():
        var_dists.at[index,"ST_Fit_30_VAR"] = res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"]))]
        var_dists.at[index,"ST_Fit_30_ES"] = np.mean(res[int(np.floor(len(res) * var_dists.loc[index, "var_perc"])):len(res)])

    return_dict["st_fit_30"] = var_dists

trials = 10000

all_data = df.get_data()

curr = all_data.loc[all_data["QTR"] == "2020-Q1",:]

tot_volume = np.sum(curr["total_volume"].to_numpy()) * 0.01

curr = curr.reset_index()

loans = int(np.floor(np.sum(curr["CURR_LOANS"]) * 0.01))

def t_inv(x, df):
    return np.sqrt(np.sum(np.power(stats.norm.rvs(size = df), 2)) / df) * stats.t.ppf(x, df = df)


loan_weight = np.zeros(loans) +  1 / loans
base_loan_corr = np.zeros(loans) + 0.15
base_loan_pd = np.zeros(loans)

low_index = 0
high_index = 0
for index, row in curr.iterrows():
    high_index += min(int(np.floor(row["CURR_LOANS"] * .01) - 1), loans)
    
    base_loan_pd[low_index:high_index] = np.mean(curr.loc[np.logical_and(all_data["CREDIT_BUCKET"] == row["CREDIT_BUCKET"], all_data["NEW_LOAN"] == row["NEW_LOAN"]), "PD"])
    low_index += min(int(np.floor(row["CURR_LOANS"] * .01) - 1), loans)





fit_data = pd.read_csv("Probability of Default/fit_model.csv")

fit_loan_corr = np.zeros(loans) + np.max(fit_data["norm_rho"].to_numpy())
fit_loan_pd = np.zeros(loans)

low_index = 0
high_index = 0
for index, row in curr.iterrows():
    high_index += min(int(np.floor(row["CURR_LOANS"] * .01) - 1), loans)

    fit_loan_pd[low_index:high_index] = fit_data.loc[np.logical_and(fit_data["CREDIT_BUCKET"] == row["CREDIT_BUCKET"], fit_data["NEW_LOAN"] == row["NEW_LOAN"]),"norm_PD"]
    low_index += min(int(np.floor(row["CURR_LOANS"] * .01) - 1), loans)

if __name__ == "__main__":

    

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []

    p = multiprocessing.Process(target = gaus_base, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_base_3, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_base_10, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_base_30, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = gaus_fit, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_fit_3, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_fit_10, args=(return_dict,))
    jobs.append(p)
    p.start()

    p = multiprocessing.Process(target = st_fit_30, args=(return_dict,))
    jobs.append(p)
    p.start()

    for proc in jobs:
        proc.join()

    var_df = return_dict["gauss_base"].join(return_dict["st_base_3"].set_index("var_perc"), on = "var_perc").join(return_dict["st_base_10"].set_index("var_perc"), on = "var_perc") \
            .join(return_dict["st_base_30"].set_index("var_perc"), on = "var_perc").join(return_dict["gauss_fit"].set_index("var_perc"), on = "var_perc").join(return_dict["st_fit_3"].set_index("var_perc"), on = "var_perc") \
                .join(return_dict["st_fit_10"].set_index("var_perc"), on = "var_perc").join(return_dict["st_fit_30"].set_index("var_perc"), on = "var_perc")


    plt.figure(figsize=(10,7.5))
    plt.plot(var_df["var_perc"], var_df["Gaussian_Base_VAR"], label = "Base Gaussian")
    plt.plot(var_df["var_perc"], var_df["ST_Base_3_VAR"], label = "Base Student-t, " + r"$\nu = 3$")
    plt.plot(var_df["var_perc"], var_df["ST_Base_10_VAR"], label = "Base Student-t, " + r"$\nu = 10$")
    plt.plot(var_df["var_perc"], var_df["ST_Base_30_VAR"], label = "Base Student-t, " + r"$\nu = 30$")
    plt.legend()
    plt.title("Base VAR Dependence Sensitivity")
    plt.xlabel("VAR")
    plt.ylabel("Percent of Portfolio")
    #Save and Clear
    plt.savefig(r"Model Results\VAR.ES\BASE_VAR.png", dpi = 600)
    plt.cla()

    plt.figure(figsize=(10,7.5))
    plt.plot(var_df["var_perc"], var_df["Gaussian_Base_ES"], label = "Base Gaussian")
    plt.plot(var_df["var_perc"], var_df["ST_Base_3_ES"], label = "Base Student-t, " + r"$\nu = 3$")
    plt.plot(var_df["var_perc"], var_df["ST_Base_10_ES"], label = "Base Student-t, " + r"$\nu = 10$")
    plt.plot(var_df["var_perc"], var_df["ST_Base_30_ES"], label = "Base Student-t, " + r"$\nu = 30$")
    plt.legend()
    plt.title("Base ES Dependence Sensitivity")
    plt.xlabel("ES")
    plt.ylabel("Percent of Portfolio")
    #Save and Clear
    plt.savefig(r"Model Results\VAR.ES\BASE_ES.png", dpi = 600)
    plt.cla()




    plt.figure(figsize=(10,7.5))
    plt.plot(var_df["var_perc"], var_df["Gaussian_Fit_VAR"], label = "Fit Gaussian")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_3_VAR"], label = "Fit Student-t, " + r"$\nu = 3$")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_10_VAR"], label = "Fit Student-t, " + r"$\nu = 10$")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_30_VAR"], label = "Fit Student-t, " + r"$\nu = 30$")
    plt.legend()
    plt.title("Fit VAR Dependence Sensitivity")
    plt.xlabel("VAR")
    plt.ylabel("Percent of Portfolio")
    #Save and Clear
    plt.savefig(r"Model Results\VAR.ES\FIT_VAR.png", dpi = 600)
    plt.cla()

    plt.figure(figsize=(10,7.5))
    plt.plot(var_df["var_perc"], var_df["Gaussian_Fit_ES"], label = "Fit Gaussian")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_3_ES"], label = "Fit Student-t, " + r"$\nu = 3$")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_10_ES"], label = "Fit Student-t, " + r"$\nu = 10$")
    plt.plot(var_df["var_perc"], var_df["ST_Fit_30_ES"], label = "Fit Student-t, " + r"$\nu = 30$")
    plt.legend()
    plt.title("Fit ES Dependence Sensitivity")
    plt.xlabel("ES")
    plt.ylabel("Percent of Portfolio")
    #Save and Clear
    plt.savefig(r"Model Results\VAR.ES\FIT_ES.png", dpi = 600)
    plt.cla()