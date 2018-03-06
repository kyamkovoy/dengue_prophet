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


for j in range(1, 11):  # want to predict at 10 time intervals, from one year forward to 10 years forward
    how_many_years_forward = j
    how_many_months = 12 * how_many_years_forward

    for prov in province_codes:
        for k in range(0, 11 - how_many_years_forward):
            print()
            print('forecasting ' + str(j) + ' year(s) forward for province ' + str(prov) + ' for year ' + str(2006 + k + j))
            print()

            df = data_file[k] # predicting for year 2007 + index

            monthly_avg = []
            for i in range(1, 13):
                monthly_cases = np.array(df['cases'].loc[df['month'] == i])
                avg = np.mean(monthly_cases)
                monthly_avg.append(avg)

            year_total = sum(monthly_avg)
            year_peak = max(monthly_avg)
            peak_month = monthly_avg.index(year_peak) + 1

            all_targets = {'year_total': year_total, 'year_peak': year_peak, 'peak_month': peak_month, 'month_cases': monthly_avg}


            # save everything
            all_folder_name = "../../output/hist_avg/all_prov_monthly/all_prov_monthly_" + str(how_many_years_forward)

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
