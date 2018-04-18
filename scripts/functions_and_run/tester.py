import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from fbprophet import Prophet
import pickle
import os
import warnings
from functions import *
warnings.simplefilter(action='ignore')

data_path = '../../data/province-month.csv'
data = get_province_data(data_path)

run_prophet(data)
