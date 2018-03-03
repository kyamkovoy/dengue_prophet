import pandas as pd
import numpy as np

data = pd.read_csv('../../data/district-biweek-counts.csv')
names = pd.read_csv('../../data/SubDistrictThailand.csv')


def get_district_codes(provinces):
    # get province codes
    provinces_upper = [province.upper() for province in provinces]
    prov_iso = names.loc[names['Province_Name'].isin(provinces_upper), ['Province_Name', 'Province_code']]
    province_codes = prov_iso.Province_code.unique()
    prov_code_dict = dict(zip(prov_iso.Province_Name, prov_iso.Province_code))

    # get district codes
    dist_iso = data.loc[data['province'].isin(province_codes), ['province', 'district']]
    district_codes = dist_iso.district.unique()
    return district_codes


def data_setup(input_data):
    provinces = ['Bangkok', 'Songkhla', 'Chiang Mai', 'Udon Thani', 'Ratchaburi']
    district_codes = get_district_codes(provinces)

    date_sick = input_data['date_sick'].loc[input_data['district'] == '1000']
    year = input_data['year'].loc[input_data['district'] == '1000']
    biweek = input_data['biweek'].loc[input_data['district'] == '1000']
    biweek_float = year + biweek/26

    cases_all_dist = []
    for district in district_codes:
        case_list = input_data['cases'].loc[input_data['district'] == district].tolist()
        cases_all_dist.append(case_list)

    case_array = np.array(cases_all_dist)

    data_dict = {'district': district_codes, 'case_matrix': case_array, 'date': date_sick, 'biweek_float': biweek_float}

    return data_dict


# dic = data_setup(data)
# matrix = dic["case_matrix"]

# np.savetxt("../output/incidence_matrix.txt", matrix, delimiter=",")

# print(data_setup(data)['district'])
