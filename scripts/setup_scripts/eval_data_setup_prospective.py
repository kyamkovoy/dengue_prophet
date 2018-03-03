import pandas as pd
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore')

data = pd.read_csv('../../data/province-biweek-counts.csv')
province_codes = [10, 41, 50, 70, 90]
years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]

# use just data for the five provinces
# make a list of dataframes with missing years
smaller_data = data.loc[data['province'].isin(province_codes)]

# make dfs of data, prospectively, train on a few years, predict the next
cv_df_list = []
for year in years:
    df = smaller_data.loc[smaller_data['year'] < year]
    cv_df_list.append(df)

cv_df_list = cv_df_list[1:]

with open('../../output/cv_df_list_prospective.pkl', 'wb') as file:
    pickle.dump(cv_df_list, file)
