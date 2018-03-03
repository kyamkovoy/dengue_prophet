import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import pickle
import os
import time
import warnings
warnings.simplefilter(action='ignore')

# this script gets monthly forecasts using biweekly data
# to compare to the monthly forecasts from monthly data

start_time = time.time()

province_codes = [10, 41, 50, 70, 90]
with open('../../output/cv_df_list_prospective.pkl', 'rb') as file:
    data_file = pickle.load(file)


# -------------------------------------------------------------------------------------------------------------------------
for j in range(1, 11):  # want to predict at 10 time intervals, from one year forward to 10 years forward
    how_many_years_forward = j
    how_many_months = 12 * how_many_years_forward

    for prov in province_codes:
        for k in range(0, 8 - how_many_years_forward):
            print()
            print('forecasting ' + str(j) + ' year(s) forward for province ' + str(prov) + ' for year ' + str(2006 + k + j))
            print()

            input_data = data_file[k] # predicting for year 2007 + index

            # setup the data
            date_sick = input_data['date_sick'].loc[input_data['province'] == 10]
            year = input_data['year'].loc[input_data['province'] == 10]
            biweek = input_data['biweek'].loc[input_data['province'] == 10]
            biweek_float = year + biweek/26

            cases_one_prov = input_data['cases'].loc[input_data['province'] == prov].tolist()  # CHANGE PROVINCE HERE
            cases = np.array(cases_one_prov)  # an array of the cases for the one province

# -------------------------------------------------------------------------------------------------------------------------
            # smooth the data
            cum_sum = cases.cumsum()
            spline = UnivariateSpline(x=biweek_float, y=cum_sum)  # try with s=0
            smooth_sum = spline(biweek_float)

            smooth_cases = np.diff(smooth_sum)
            x = np.append(smooth_cases[::-1], cum_sum[0])
            smooth_cases = x[::-1]  # smoothed case counts for the one province

# ------------------------------------------------------------------------------------------------------------------------
            # prepare dataframe for prophet
            df = pd.DataFrame(list(zip(date_sick, cases)), columns=['ds', 'y'])

            # fit the prophet model
            model = Prophet(changepoint_prior_scale=0.5, interval_width=0.95, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df)

            # extend the dataframe and get the predictions
            future_dates = model.make_future_dataframe(periods=12, freq='M')
            past_and_forecast = model.predict(future_dates)
            forecast = past_and_forecast.tail(12)

# -------------------------------------------------------------------------------------------------------------------------
            '''
            # forecast is in months, need to interpolate to get biweeks
            forecast['yhat_sum'] = forecast['yhat'].cumsum()
            x = np.linspace(1/12, 1, 12)
            fore_spline = UnivariateSpline(x=x, y=forecast['yhat_sum'], s=0)

            xhat = np.linspace(1/26, 1, 26)           # biweek float
            forecast_biweeks_sum = fore_spline(xhat)  # cumulative predictions

            biweek_cases = np.diff(forecast_biweeks_sum)
            x = np.append(biweek_cases[::-1], forecast_biweeks_sum[0])
            biweek_cases = x[::-1]

            # get the targets, biweek cases is the biweekly target
            biweek_cases = biweek_cases.tolist()
            year_total = sum(biweek_cases)
            year_peak = max(biweek_cases)
            peak_biweek = biweek_cases.index(year_peak)

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_biweek': peak_biweek, 'biweek_cases': biweek_cases}
'''
# ------------------------------------------------------------------------------------------------------------------------
            # get MONTHLY targets
            month_cases = forecast['yhat'].tolist()
            year_total = sum(month_cases)
            year_peak = max(month_cases)
            peak_month = month_cases.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': month_cases}

# ------------------------------------------------------------------------------------------------------------------------
            # save everything
            all_folder_name = "../../output/all_prov_biweekly/all_prov_biweekly_" + str(how_many_years_forward)

            if not os.path.exists(all_folder_name):
                os.makedirs(all_folder_name)

            folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_biweekly_" + str(how_many_years_forward)
            file_name = "prov_" + str(prov) + "_for_" + str(2006 + k + j) + "_biweekly_" + str(how_many_years_forward) + ".pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print()
            print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

            with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
                pickle.dump(all_targets, file)


print(time.time() - start_time)
