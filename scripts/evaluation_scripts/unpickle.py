import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import warnings
warnings.simplefilter(action='ignore')


with open('../../output/all_prov_monthly/errors.pkl', 'rb') as file:
    output = pickle.load(file)

print(output)
