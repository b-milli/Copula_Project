

import numpy as np
import pandas as pd


def get_data():

    in_data = pd.read_csv(r"Probability of Default/use_data.csv")

    return in_data