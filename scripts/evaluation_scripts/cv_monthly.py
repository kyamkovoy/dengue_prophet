import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error as mse
import os
import warnings
warnings.simplefilter(action='ignore')

data =  pd.read_csv('../../data/province-month.csv')

province_codes = [10, 41, 50, 70, 90]

which_model = 'cp_prophet'
which_time0 = 'jul_time0'


folder_name = '../../output/' + which_time0 + '/' + which_model
cv_df = pd.DataFrame(columns=['province', 'year', 'monthly_case_rmse', 'year_total_error', 'peak_error', 'peak_month_error'])

for prov in province_codes:
    for k in range(0, 10):

        year = 2007 + k

        print()
        print('province ' + str(prov) + ': year ' + str(year))

        file_name = '/prov_' + str(prov) + '_monthly/prov_' + str(prov) + '_for_' + str(year) + '_monthly.pkl'

        with open(folder_name + file_name, 'rb') as file:
            forecast = pickle.load(file)

        true_df = data.loc[(data['province'] == prov) & (data['date_sick_year'] == year)]

        month_cases = true_df['cases'].tolist()
        total = sum(month_cases)
        peak = max(month_cases)
        peak_month = month_cases.index(peak) + 1

        if len(month_cases) == len(forecast['month_cases']):
            case_rmse = np.sqrt(mse(month_cases, forecast['month_cases']))
        else:
            case_rmse = 'NaN'
        year_total_error = abs(total - forecast['year_total'])
        peak_error = abs(peak - forecast['year_peak'])
        peak_month_error = abs(peak_month - forecast['peak_month'])

        row = [prov, year, case_rmse, year_total_error, peak_error, peak_month_error]

        cv_df = cv_df.append(pd.Series(row, index=cv_df.columns), ignore_index = True)


cv_df.to_csv(folder_name + '/' + 'monthly_errors.csv', index=False)
