import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import os
import warnings
from prophet_functions import *
from avg_monthly_functions import *
warnings.simplefilter(action='ignore')

data_path = '../../data/province-month.csv'
data = get_province_data(data_path)

run_avg_model(data)
