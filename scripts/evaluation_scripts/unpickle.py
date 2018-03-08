import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import warnings
warnings.simplefilter(action='ignore')


with open('../../output/hist_avg/all_prov_biweekly/all_prov_biweekly_1/prov_10_biweekly_1/prov_10_for_2007_biweekly_1.pkl', 'rb') as file:
    output = pickle.load(file)

print(output)
