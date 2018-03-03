import pandas as pd
import numpy as np

data = pd.read_csv('../../../data/province-biweek-counts.csv')
province_codes = [10, 41, 50, 70, 90]

def data_setup(input_data):
    date_sick = input_data['date_sick'].loc[input_data['province'] == 10]
    year = input_data['year'].loc[input_data['province'] == 10]
    biweek = input_data['biweek'].loc[input_data['province'] == 10]
    biweek_float = year + biweek/26

    cases_all_prov = []
    for province in province_codes:
        case_list = input_data['cases'].loc[input_data['province'] == province].tolist()
        cases_all_prov.append(case_list)

    case_array = np.array(cases_all_prov)

    data_dict = {'province': province_codes, 'case_matrix': case_array, 'date': date_sick, 'biweek_float': biweek_float}

    return data_dict
