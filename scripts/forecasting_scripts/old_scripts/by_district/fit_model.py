import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
from preprocessing import *
import warnings
warnings.simplefilter(action='ignore')


data = pd.read_csv('../../data/district-biweek-counts.csv')
names = pd.read_csv('../../data/spatial-files-from-NECTEC/SubDistrictThailand.csv')

provinces = ['Bangkok', 'Songkhla', 'Chiang Mai', 'Udon Thani', 'Ratchaburi']

# -------------------------------------------------------------------------------------------------------


def smooth_past_data(input_data):  # this will call the data setup function from preprocessing
    data_dict = data_setup(input_data)
    district_codes = data_dict['district']
    cases = data_dict['case_matrix']
    date = data_dict['date']
    biweek = data_dict['biweek_float']

    smooth_case_list = []
    for row in range(0, 125):
        print('smoothing data for district ' + str(data_dict['district'][row]))

        cum_sum = cases[row].cumsum()
        spline = UnivariateSpline(x=biweek, y=cum_sum)  # try with s=0
        smooth_sum = spline(biweek)

        smooth_cases = np.diff(smooth_sum)
        x = np.append(smooth_cases[::-1], cum_sum[0])
        smooth_cases = x[::-1]

        smooth_case_list.append(smooth_cases)

    smooth_case_matrix = np.array(smooth_case_list)
    smooth_data_dict = {'district': district_codes, 'smooth_case_matrix': smooth_case_matrix, 'date': date, 'biweek_float': biweek}

    return smooth_data_dict


# get the fitted prophet model, returns models in a list in order of district
def fit_prophet_model(input_data):
    data_dict = smooth_past_data(input_data)
    date = data_dict['date']
    cases = data_dict['smooth_case_matrix']
    district = data_dict['district']

    model_list = []
    for i in range(0, len(district)):
        print("fitting model for district " + district[i])

        # prepare dataframe for prophet
        df = pd.DataFrame(list(zip(date, cases[i])), columns=['ds', 'y'])

        # fit the prophet model
        model = Prophet(interval_width=0.95, mcmc_samples=10)  # mcmc_samples = 300? start with 10
        model.fit(df)

        model_list.append(model)

    return model_list

