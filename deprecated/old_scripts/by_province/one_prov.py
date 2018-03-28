import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import pickle
from preprocessing_prov import *
from fit_model_prov import *
import warnings
warnings.simplefilter(action='ignore')

# input_data = pd.read_csv('../../../data/province-biweek-counts.csv')
# province_codes = [10, 41, 50, 70, 90]

print('unpickling...')

with open('../../../output/cv_df_list_prospective_monthly.pkl', 'rb') as file:
    data_file = pickle.load(file)

prov = 90  # CHANGE PROVINCE HERE

# ------------------------------------------------------------------------------

for k in range(0, 7):
    input_data = data_file[k] # predicting for year 2007 + index

    print()
    print('setting up data...')

    date_sick = input_data['date_sick'].loc[input_data['province'] == 10]
    year = input_data['date_sick_year'].loc[input_data['province'] == 10]
    biweek = input_data['biweek'].loc[input_data['province'] == 10]
    biweek_float = year + biweek/26

    cases_one_prov = input_data['cases'].loc[input_data['province'] == prov].tolist()  # CHANGE PROVINCE HERE
    cases = np.array(cases_one_prov)  # an array of the cases for the one province

    # ------------------------------------------------------------------------------
    print()
    print('smoothing data...')

    cum_sum = cases.cumsum()
    spline = UnivariateSpline(x=biweek_float, y=cum_sum)  # try with s=0
    smooth_sum = spline(biweek_float)

    smooth_cases = np.diff(smooth_sum)
    x = np.append(smooth_cases[::-1], cum_sum[0])
    smooth_cases = x[::-1]  # smoothed case counts for the one province

    # ------------------------------------------------------------------------------
    print()
    print("fitting model...")

    # prepare dataframe for prophet
    df = pd.DataFrame(list(zip(date_sick, cases)), columns=['ds', 'y'])

    # fit the prophet model
    model = Prophet(interval_width=0.95)  # mcmc_samples = 300? start with 10
    model.fit(df)

    # ------------------------------------------------------------------------------
    print()
    print("getting targets...")

    future_dates = model.make_future_dataframe(periods=12, freq='M')
    past_and_forecast = model.predict(future_dates)
    forecast = past_and_forecast.tail(12)

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
    year_total = forecast_biweeks_sum[len(forecast_biweeks_sum)-1]
    year_peak = max(biweek_cases)

    peak_biweek = 0
    for i in range(0, len(biweek_cases)):
        if biweek_cases[i] == year_peak:
            peak_biweek += i + 1

    # probability = model.predictive_samples(future_dates)
    all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_biweek': peak_biweek, 'biweek_cases': biweek_cases}

    all_folder_name = "../../../output/all_prov_monthly"

    if not os.path.exists(all_folder_name):
        os.makedirs(all_folder_name)

    folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_monthly"
    file_name = "prov_" + str(prov) + "_for_" + str(2007 + k) + "_monthly.pkl"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print()
    print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE


    # missing the year in the file name
    with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
        pickle.dump(all_targets, file)
