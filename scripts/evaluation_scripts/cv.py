import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.simplefilter(action='ignore')

data =  pd.read_csv('../../data/province-biweek-counts.csv')

with open('../../output/all_prov_no_prob/prov_10_no_prob/prov_10_for_2013.pkl', 'rb') as file:
    forecast = pickle.load(file)

province = 10
year = 2013

true_df = data.loc[(data['province'] == province) & (data['year'] == year)]

biweek_cases = true_df['cases'].tolist()
total = sum(biweek_cases)
peak = max(biweek_cases)
peak_biweek = biweek_cases.index(peak) + 1

biweek_rmse = np.sqrt(mse(biweek_cases, forecast['biweek_cases']))
print(biweek_rmse)

year_total_rmse = np.sqrt((total - forecast['year_total'])**2)
print(forecast['year_total'])
print(year_total_rmse)
