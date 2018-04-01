import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import warnings
warnings.simplefilter(action='ignore')


with open('../../output/cv_df_list_prospective_monthly_july.pkl', 'rb') as file:
    output = pickle.load(file)

print(output)
