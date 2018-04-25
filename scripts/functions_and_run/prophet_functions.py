import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')

# use this to get the monthly predictions starting at every month

data_path = '../../data/province-month.csv'

#------------------------------------------------------------------------------------------------------------------------------

def get_province_data(data_path):

    province_codes = [10, 41, 50, 70, 90]
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

    data = pd.read_csv(data_path)
    smaller_data = data.loc[data['province'].isin(province_codes) & data['date_sick_year'].isin(years)]

    # convert the year and month into a datetime (called date_sick)
    smaller_data['date_sick_year'].apply(str)
    smaller_data['month'].apply(str)

    smaller_data['date_sick'] = smaller_data['date_sick_year'].map(str) + '-' + smaller_data['month'].map(str) + '-' + str(1)

    return smaller_data


#----------------------------------------------------------------------------------------------------------------------------

def setup_data(data, province, year, month):  # year and month being predicted for
    # years: 2006-2016, forecast for 2007+ starting in every month

    prov_data = data.loc[(data['province'] == province) & (data['date_sick_year'] < (year + 1))]
    cutoff = 13 - month
    df = prov_data.head(prov_data.shape[0]-cutoff)

    date_sick = df['date_sick']
    cases = np.array(df['cases'])

    # prepare dataframe for prophet
    df = pd.DataFrame(list(zip(date_sick, cases)), columns=['ds', 'y'])

    return df


#--------------------------------------------------------------------------------------------------------------------------------------------

def run_prophet(data):

    province_codes = [10, 41, 50, 70, 90]
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

    for prov in province_codes:
        for year in range(2007, 2017):
            for month in range(1, 13):

                df = setup_data(data, prov, year, month)

                print()
                print('forecasting...')
                print('province: ' + str(prov) + ', year: ' + str(year) + ', month: ' + str(month))
                print()

                model = Prophet(interval_width=0.95, changepoint_prior_scale=0.5, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(df)

                future_dates = model.make_future_dataframe(periods=12, freq='M')
                past_and_forecast = model.predict(future_dates)
                forecast = past_and_forecast.tail(12)

                all_folder_name = "../../output/monthly_forecasts/flexible_prophet/"

                if not os.path.exists(all_folder_name):
                    os.makedirs(all_folder_name)

                folder_name = all_folder_name + '/prov_' + str(prov) + '_monthly/'
                file_name = 'prov_' + str(prov) + '_' + str(year) + '_' + str(month) + '_monthly.csv'

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                print()
                print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

                forecast.to_csv(folder_name + file_name, index=False)

    print()
    print('done')
    print()
