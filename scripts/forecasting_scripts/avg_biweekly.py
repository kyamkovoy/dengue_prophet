import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')

province_codes = [10, 41, 50, 70, 90]

with open('../../output/cv_df_list_prospective.pkl', 'rb') as file:
    data_file = pickle.load(file)


def get_monthly_avg(df):
    # gets average values for each month in the df
    df['year_biweek'] = (df['year']-2006)*26 + (df['biweek'])

    # want to first smooth the data
    x = np.array(df['year_biweek'])
    y = np.array(df['cases'])
    y_sum = np.cumsum(y)

    spl = UnivariateSpline(x, y_sum)
    y_fit = spl(x)
    y_smooth = np.diff(y_fit)
    n = np.append(y_smooth[::-1], y_sum[0])
    y_smooth = n[::-1]
    df['smooth_cases'] = y_smooth

    df['date_sick'] = pd.to_datetime(df['date_sick'])
    df['month'] = 0
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    df_w_month = pd.DataFrame(columns = ['province', 'year', 'biweek', 'date_sick', 'cases', 'year_biweek', 'smooth_cases', 'month'])
    for year in years:
        for i in range(1, 12):
            month_start = str(year) + '-' + str(i)
            month_end = str(year) + '-' + str(i+1)
            date_range = (df['date_sick'] >= month_start) & (df['date_sick'] < month_end)
            one_month = df.loc[date_range]
            one_month['month'] = i
            df_w_month = df_w_month.append(one_month)


    # then get the averages
    monthly_avg = []
    for i in range(1, 13):
        monthly_cases = df_w_month['smooth_cases'].loc[df_w_month['month'] == i].tolist()
        avg = np.mean(monthly_cases)
        monthly_avg.append(avg)

    return monthly_avg



for j in range(1, 11):  # want to predict at 10 time intervals, from one year forward to 10 years forward
    how_many_years_forward = j
    how_many_months = 12 * how_many_years_forward

    for prov in province_codes:
        for k in range(0, 8 - how_many_years_forward):
            print()
            print('forecasting ' + str(j) + ' year(s) forward for province ' + str(prov) + ' for year ' + str(2006 + k + j))
            print()

            df = data_file[k] # predicting for year 2007 + index

            monthly_avg = get_monthly_avg(df)

            year_total = sum(monthly_avg)
            year_peak = max(monthly_avg)
            peak_month = monthly_avg.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': monthly_avg}


            # save everything
            all_folder_name = "../../output/hist_avg/all_prov_biweekly/all_prov_biweekly_" + str(how_many_years_forward)

            if not os.path.exists(all_folder_name):
                os.makedirs(all_folder_name)

            folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_biweekly_" + str(how_many_years_forward)
            file_name = "prov_" + str(prov) + "_for_" + str(2006 + k + j) + "_biweekly_" + str(how_many_years_forward) + ".pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print()
            print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

            with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
                pickle.dump(all_targets, file)
