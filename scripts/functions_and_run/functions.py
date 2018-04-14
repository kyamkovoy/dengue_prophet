import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')

data = '../../data/province-month.csv'


def get_province_data(data_path):

    data = pd.read_csv(data_path)
    smaller_data = data.loc[data['province'].isin(province_codes) & data['date_sick_year'].isin(years)]

    # convert the year and month into a datetime (called date_sick)
    smaller_data['date_sick_year'].apply(str)
    smaller_data['month'].apply(str)

    smaller_data['date_sick'] = smaller_data['date_sick_year'].map(str) + '-' + smaller_data['month'].map(str) + '-' + str(1)

    return smaller_data


def setup_data(data):

    cv_df_list = []
    for year in years:
        df = smaller_data.loc[smaller_data['date_sick_year'] < year]
        cv_df_list.append(df)

    cv_df_list = cv_df_list[1:]



def run_prophet(data_path, time_zero, growth):

    province_codes = [10, 41, 50, 70, 90]
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]



    for prov in province_codes:
        for k in range(0, 10):
            print()
            print('forecasting for province ' + str(prov) + ' for year ' + str(2007 + k))
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
            future_dates = model.make_future_dataframe(periods=12, freq='M')
            past_and_forecast = model.predict(future_dates)
            forecast = past_and_forecast.tail(12)


            # get the targets
            month_cases = forecast['yhat'].tolist()
            year_total = sum(month_cases)
            year_peak = max(month_cases)
            peak_month = month_cases.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': month_cases}


            # save everything
            all_folder_name = "../../output/jun_time0/default_prophet/"

            if not os.path.exists(all_folder_name):
                os.makedirs(all_folder_name)

            folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_monthly"
            file_name = "prov_" + str(prov) + "_for_" + str(2007 + k) + "_monthly.pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print()
            print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

            with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
                pickle.dump(all_targets, file)
