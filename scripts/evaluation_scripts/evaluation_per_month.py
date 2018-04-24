import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error as mse
import os
import warnings
warnings.simplefilter(action='ignore')

data =  pd.read_csv('../../data/province-month.csv')
province_codes = [10]

which_model = 'default_prophet'

folder_name = '../../output/monthly_forecasts/' + which_model + '/'

# -----------------------------------------------------------------------------------------------------------------

for prov in province_codes:
    prov_folder = folder_name + 'prov_' + str(prov) + '_monthly/'
    prov_data = data.loc[data['province'] == prov]

    for year in range(2007, 2017):
        for month in range(1, 13):
            # load the file for forecasts starting in month, year and get forecasts into an array
            file_name = prov_folder + 'prov_' + str(prov) + '_' + str(year) + '_' + str(month) + '_monthly.csv'

            print(file_name)

            forecast_df = pd.read_csv(file_name)
            forecast_cases = np.array(forecast_df['yhat'])

            # list of months in order, ie if month = 3, list starts at 3, ends at 2, contains 12 months
            months_first = np.arange(month, 13)
            months_last = np.arange(1, month)
            month_list = np.concatenate([months_first, months_last])

            # do the same thing with years
            year_list = [year if i >=month else year + 1 for i in month_list]

            # now get the list of true cases
            true_cases = []
            for i in range(0, 12):
                one_year = year_list[i]
                one_month = month_list[i]
                print(one_year)
                print(one_month)
                blah = prov_data['cases'].loc[(prov_data['date_sick_year'] == one_year) &  (prov_data['month'] == one_month)]

                print(blah)
                print(type(blah))

                cases_df = 0
                if blah.shape[0] == 1:
                    cases_df = float(blah)
                else:
                    cases_df = 0.

                true_cases.append(cases_df)

            true_cases = np.array(true_cases)

            # get the errors and put everything into a dataframe
            errors = abs(true_cases - forecast_cases)

            error_dict = {'year': year_list, 'month': month_list, 'error': errors}
            error_df = pd.DataFrame(error_dict)

            save_file = prov_folder + 'prov_' + str(prov) + '_' + str(year) + '_' + str(month) + '_monthly_errors.csv'
            error_df.to_csv(save_file, index=False)

            print(true_cases)
            print(forecast_cases)
