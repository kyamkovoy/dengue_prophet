import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import pickle
import os
import csv
import warnings
warnings.simplefilter(action='ignore')

province_codes = [10, 41, 50, 70, 90]

data_path = '../../data/province-month.csv'

def setup_data(data, province, year, month):  # year and month being predicted for
    # years: 2006-2016, forecast for 2007+ starting in every month

    prov_data = data.loc[(data['province'] == province) & (data['date_sick_year'] < (year + 1))]
    cutoff = 13 - month
    df = prov_data.head(prov_data.shape[0]-cutoff)

    return df


def get_monthly_avg(df):
    # gets average values for each month in the df
    df['year_month'] = (df['date_sick_year']-2006)*12 + (df['month'])

    # want to first smooth the data
    x = np.array(df['year_month'])
    y = np.array(df['cases'])
    y_sum = np.cumsum(y)
    spl = UnivariateSpline(x, y_sum)
    y_fit = spl(x)
    y_smooth = np.diff(y_fit)
    n = np.append(y_smooth[::-1], y_sum[0])
    y_smooth = n[::-1]
    df['smooth_cases'] = y_smooth

    # then get the averages
    monthly_avg = []
    for i in range(1, 13):
        monthly_cases = np.array(df['smooth_cases'].loc[df['month'] == i])
        avg = np.mean(monthly_cases)
        monthly_avg.append(avg)

    return monthly_avg


def run_avg_model(data):

    province_codes = [10, 41, 50, 70, 90]
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

    for prov in province_codes:
        for year in range(2007, 2017):
            for month in range(1, 13):

                df = setup_data(data, prov, year, month)
                print(df.head())

                print()
                print('forecasting...')
                print('province: ' + str(prov) + ', year: ' + str(year) + ', month: ' + str(month))
                print()

                forecast = get_monthly_avg(df)
                forecast_df = pd.DataFrame({'forecast': forecast})

                all_folder_name = "../../output/monthly_forecasts/hist_avg/"

                if not os.path.exists(all_folder_name):
                    os.makedirs(all_folder_name)

                folder_name = all_folder_name + '/prov_' + str(prov) + '_monthly/'
                file_name = 'prov_' + str(prov) + '_' + str(year) + '_' + str(month) + '_monthly.csv'

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                print()
                print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

                forecast_df.to_csv(folder_name + file_name, index=False)

    print()
    print('done')
    print()
