import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import pickle
import os
import warnings
warnings.simplefilter(action='ignore')

province_codes = [10, 41, 50, 70, 90]

with open('../../output/cv_df_list_prospective_monthly.pkl', 'rb') as file:
    data_file = pickle.load(file)


def get_monthly_avg(df):
    # gets average values for each month in the df
    df['year_month'] = (df['date_sick_year']-2006)*12 + (df['month'])

    # want to first smooth the data
    x = np.array(df['year_month'])
    y = np.array(df['cases'])
    y_sum = np.cumsum(y)
    spl = UnivariateSpline(x, y_sum)
    y_fit = spl(x)
    y_smooth = np.diff(y_fit)
    n = np.append(y_smooth[::-1], y_sum[0])
    y_smooth = n[::-1]
    df['smooth_cases'] = y_smooth

    # then get the averages
    monthly_avg = []
    for i in range(1, 13):
        monthly_cases = np.array(df['smooth_cases'].loc[df['month'] == i])
        avg = np.mean(monthly_cases)
        monthly_avg.append(avg)

    return monthly_avg


for prov in province_codes:

    # past_predictions = []

    for j in range(1, 11):  # want to predict at 10 time intervals, from one year forward to 10 years forward
        how_many_years_forward = j
        how_many_months = 12 * how_many_years_forward

        for k in range(0, 11 - how_many_years_forward):
            print()
            print('forecasting ' + str(j) + ' year(s) forward for province ' + str(prov) + ' for year ' + str(2006 + k + j))
            print()

            df = data_file[k] # predicting for year 2007 + index
            df = df.loc[df['province'] == prov]

            # get the monthly avg values
            monthly_avg = get_monthly_avg(df)

            year_total = sum(monthly_avg)
            year_peak = max(monthly_avg)
            peak_month = monthly_avg.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': monthly_avg}


            # save everything
            all_folder_name = "../../output/hist_avg_test/all_prov_monthly/all_prov_monthly_" + str(how_many_years_forward)

            if not os.path.exists(all_folder_name):
                os.makedirs(all_folder_name)

            folder_name = all_folder_name + "/" + "prov_" + str(prov) + "_monthly_" + str(how_many_years_forward)
            file_name = "prov_" + str(prov) + "_for_" + str(2006 + k + j) + "_monthly_" + str(how_many_years_forward) + ".pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print()
            print("saving output to " + file_name)  # CHANGE PROV AND YEAR HERE

            with open(folder_name + '/' + file_name, 'wb') as file:  # CHANGE PROV AND YEAR HERE
                pickle.dump(all_targets, file)
