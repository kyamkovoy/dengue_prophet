import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')

province_codes = [10, 41, 50, 70, 90]

with open('../../output/cv_df_list_prospective_monthly.pkl', 'rb') as file:
    data_file = pickle.load(file)

new_dfs = []
for df in data_file:
    new_df = df.head(df.shape[0] - 30)
    new_dfs.append(new_df)

with open('../../output/cv_df_list_prospective_monthly_july.pkl', 'wb') as file:
    pickle.dump(new_dfs, file)
