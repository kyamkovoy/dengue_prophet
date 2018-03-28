import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
from preprocessing_prov import *
from fit_model_prov import *
import warnings
warnings.simplefilter(action='ignore')

data = pd.read_csv('../../../data/province-biweek-counts.csv')
province_codes = [10, 41, 50, 70, 90]


def get_targets(input_data):
    data_dict = data_setup(input_data)
    model_list = fit_prophet_model(input_data)

    province = data_dict['province']
    # biweek = data_dict['biweek']

    all_forecasts = []
    j = 0
    for model in model_list:
        print("forecasting for province " + str(province[j]))

        future_dates = model.make_future_dataframe(periods=12, freq='M')
        past_and_forecast = model.predict(future_dates)
        forecast = past_and_forecast.tail(12)

        # this forecast is in months, need to interpolate to get biweeks, do same as above
        forecast['yhat_sum'] = forecast['yhat'].cumsum()
        x = np.linspace(1/12, 1, 12)
        fore_spline = UnivariateSpline(x=x, y=forecast['yhat_sum'], s=0)

        xhat = np.linspace(1/26, 1, 26)           # biweek float
        forecast_biweeks_sum = fore_spline(xhat)  # cumulative predictions
        # print(forecast_biweeks_sum)

        biweek_cases = np.diff(forecast_biweeks_sum)
        x = np.append(biweek_cases[::-1], forecast_biweeks_sum[0])
        biweek_cases = x[::-1]

        # get the targets, biweek cases is the biweekly target
        year_total = forecast_biweeks_sum[len(forecast_biweeks_sum)-1]
        year_peak = max(biweek_cases)

        peak_biweek = 0
        for i in range(0, len(biweek_cases)):
            if biweek_cases[i] == year_peak:
                peak_biweek += i + 1

        # probability = model.predictive_samples(future_dates)
        all_forecasts.append([year_total, year_peak, peak_biweek, biweek_cases])

        j += 1

    return all_forecasts


forecasts = get_targets(data)

with open('../../output/province_targets_test_1', 'wb') as file:
    pickle.dump(forecasts, file)
