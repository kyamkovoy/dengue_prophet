import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')


province_codes = [10, 41, 50, 70, 90]

with open('../../../output/cv_df_list_prospective_monthly.pkl', 'rb') as file:
    data_file = pickle.load(file)


for j in range(1, 11):  # want to predict at 10 time intervals, from one year forward to 10 years forward
    how_many_years_forward = j
    how_many_months = 12 * how_many_years_forward

    for prov in province_codes:
        for k in range(0, 11 - how_many_years_forward):
            print()
            print('forecasting ' + str(j) + ' year(s) forward for province ' + str(prov) + ' for year ' + str(2006 + k + j))
            print()

            input_data = data_file[k] # predicting for year 2007 + index

            date_sick = input_data['date_sick'].loc[input_data['province'] == 10]
            year = input_data['date_sick_year'].loc[input_data['province'] == 10]

            cases_one_prov = input_data['cases'].loc[input_data['province'] == prov].tolist()
            cases = np.array(cases_one_prov)  # an array of the cases for the one province


            # prepare dataframe for prophet
            df = pd.DataFrame(list(zip(date_sick, cases)), columns=['ds', 'y'])

            # fit the prophet model
            model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df)

            # extend the dataframe and make the prediction
            future_dates = model.make_future_dataframe(periods=how_many_months, freq='M')
            past_and_forecast = model.predict(future_dates)
            forecast = past_and_forecast.tail(12)


            # get the targets
            month_cases = forecast['yhat'].tolist()
            year_total = sum(month_cases)
            year_peak = max(month_cases)
            peak_month = month_cases.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': month_cases}


            # save everything
            all_folder_name = "../../../output/all_prov_monthly_" + str(how_many_years_forward)

            if not os.path.exists(all_folder_name):
                os.makedirs(all_folder_name)

            folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_monthly_" + str(how_many_years_forward)
            file_name = "prov_" + str(prov) + "_for_" + str(2006 + k + j) + "_monthly_" + str(how_many_years_forward) + ".pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print()
            print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

            with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
                pickle.dump(all_targets, file)
