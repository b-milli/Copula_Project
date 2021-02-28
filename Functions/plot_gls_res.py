
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_gls(data,res):


    for c in data["CREDIT_BUCKET"].unique():
        
        tmp_dte_new = data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0),["QTR2"]]
        tmp_act_new = data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0),["PD"]]
        tmp_predict_new = stats.norm.cdf(res.predict()[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 0)])

        tmp_dte_old = data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1),["QTR2"]]
        tmp_act_old = data.loc[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1),["PD"]]
        tmp_predict_old = stats.norm.cdf(res.predict()[np.logical_and(data["CREDIT_BUCKET"] == c, data["NEW_LOAN"] == 1)])

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.scatter(tmp_dte_new, tmp_act_new, label = "Default Experiance")
        ax1.plot(tmp_dte_new, tmp_predict_new, label = "Default Predicted")

        ax2.scatter(tmp_dte_old, tmp_act_old, label = "Default Experaince")
        ax2.plot(tmp_dte_old, tmp_predict_old, label = "Default Predicted")

        fig.savefig(r"./Model Results/Plots/Comparison/model_comp_"+c+".png",dpi = 600)
