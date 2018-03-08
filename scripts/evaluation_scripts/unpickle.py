import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import warnings
warnings.simplefilter(action='ignore')


with open('../../output/hist_avg/all_prov_monthly/all_prov_monthly_1/prov_41_monthly_1/prov_41_for_2007_monthly_1.pkl', 'rb') as file:
    output = pickle.load(file)

print(output)
