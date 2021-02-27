
import numpy as np
import scipy.stats as stats


def Copula_Loop(Loan_weight, loan_corr, Unconditional_Default, y_gen_function, f_inv, g_cdf, trials):

    #Assign memory for trial outcomes
    trial_default = np.zeros(trials)

    un_con_pd = f_inv(Unconditional_Default)    
    
    #Generate a random variable Y for the systematic risk
    y = y_gen_function(trials)

    #Loop over trials
    for trial in range(trials):

        #Expected Loss calcualtion
        trial_default[trial] = np.sum(Loan_weight * g_cdf((un_con_pd - np.sqrt(loan_corr) * y[trial]) / np.sqrt(1 - loan_corr)))

    return trial_default

