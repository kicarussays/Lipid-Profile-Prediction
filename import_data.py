import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def DATA_PREPROCESS():
    """
    Data Preprocessing
    """

    data = pd.read_csv('AI_DATA_SORT.csv')
    size = data.shape

    missing = data.isnull().sum()

    data = data.dropna(subset=['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl'])

    # Include feature having # of Nan less than 20%
    data = data[missing[missing < 0.2 * size[0]].index].dropna(axis=0)

    # 0 for Female, 1 for Male
    data['SEX_TP_CD'] = data.apply(lambda row: 0 if row['SEX_TP_CD']=='F' else 1, axis=1)

    # Control date type
    data['ORD_DT'] = data['ORD_DT'].astype('datetime64[ns]')
    data['PT_BRDY_DT'] = data['PT_BRDY_DT'].astype('datetime64[ns]')
    data['AGE'] = (data['ORD_DT'] - data['PT_BRDY_DT']).dt.days / 365.25

    # Remove useless variables
    data = data[data.columns.difference(['RAND', 'RAND_KEY', 'PT_BRDY_DT', 'ORD_DT'])]

    # Remove rows with string data
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[i].astype(float)
        except ValueError as e:
            # print(e)
            remove_row.append(i)

    data = data.drop(data.index[remove_row]).astype(float)
    data['BMI'] = data['HPC00002_weight'] / (data['HPC00001_height'] / 100) ** 2
    data['WtH'] = data['HPC00007_wc'] / data['HPC00001_height']

    _size1 = data.shape


    # TG log 변환
    for i in range(_size1[0]):
        data['L3061_tg'].iloc[i] = np.log(data['L3061_tg'].iloc[i])

    data = data[['AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight',
                'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp', 'HPC00006_pulse',
                'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast',
                'L3015_alt', 'SEX_TP_CD', 'BMI', 'WtH',
                'L3008_cholesterol', 'L3061_tg', 'L3062_hdl','L3068_ldl']]

    xdata = data[data.columns.difference(['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl'])]
    ydata = data[['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    return data, xdata, ydata




def DATA_PREPROCESS_v2():
    """
    Data Preprocessing
    """

    data = pd.read_csv('hn_test.csv')
    size = data.shape

    # y 변수 조정
    data = data.dropna(subset=['HE_chol', 'HE_TG', 'HE_HDL_st2']) # LDL은 결측치에 대하여 계산 가능
    _size1 = data.shape
    _LDL_n = data['HE_LDL_drct'].isna()

    # LDL 계산 --- HE_chol - (HE_TG / (5 + HE_HDL_st2))
    for i in range(_size1[0]):
        if _LDL_n.iloc[i]:
            data['HE_LDL_drct'].iloc[i] = \
                data['HE_chol'].iloc[i] - (data['HE_TG'].iloc[i] / (5 + data['HE_HDL_st2'].iloc[i]))

    # TG log 변환
    for i in range(_size1[0]):
        data['HE_TG'].iloc[i] = np.log(data['HE_TG'].iloc[i])

    # 결측치 현황
    missing = data.isnull().sum()

    # Sex, Age / 신체계측 변수 포함
    data = data[['sex', 'age', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_obe', 'HE_sbp', 'HE_dbp',
                 'HE_chol', 'HE_TG', 'HE_HDL_st2', 'HE_LDL_drct']].dropna(axis=0)


    # Remove rows with string data
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[i].astype(float)
        except ValueError as e:
            # print(e)
            remove_row.append(i)

    data = data.drop(data.index[remove_row])
    xdata = data[data.columns.difference(['HE_chol', 'HE_TG', 'HE_HDL_st2', 'HE_LDL_drct'])]
    ydata = data[['HE_chol', 'HE_TG', 'HE_HDL_st2', 'HE_LDL_drct']]

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    return data, xdata, ydata


def DATA_PREPROCESS_SH(log=True):
    data = pd.read_csv('AI_DATA_SORT.csv')
    size = data.shape

    data = data.dropna(subset=['L3008_cholesterol', 'L3061_tg', 'L3062_hdl'])

    # 0 for Female, 1 for Male
    data['SEX_TP_CD'] = data.apply(lambda row: 0 if row['SEX_TP_CD'] == 'F' else 1, axis=1)

    # Control date type
    data['ORD_DT'] = data.RAND_KEY.str[-10:]
    data['ORD_DT'] = data['ORD_DT'].astype('datetime64[ns]')
    data['PT_BRDY_DT'] = data['PT_BRDY_DT'].astype('datetime64[ns]')
    data['AGE'] = (data['ORD_DT'] - data['PT_BRDY_DT']).dt.days / 365.25

    # age >= 20
    data = data.loc[data.AGE >= 20, :]
    # Remove useless variables
    data = data[data.columns.difference(['RAND', 'PT_BRDY_DT', 'ORD_DT'])]

    # Remove rows with string data
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43]].astype(float)
        except ValueError as e:
            remove_row.append(i)

    data = data.drop(data.index[remove_row])
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43]:
        name = data.columns[i]
        data = data.astype({name: 'float'})

    data['BMI'] = data['HPC00002_weight'] / (data['HPC00001_height'] / 100) ** 2
    data['WtH'] = data['HPC00007_wc'] / data['HPC00001_height']

    # ldl imputation
    data['L3068_ldl'] = np.where(pd.notnull(data['L3068_ldl']) == True, data['L3068_ldl'],
                                 data['L3008_cholesterol'] - ((data['L3061_tg'] / 5) + data['L3062_hdl']))

    # TG log 변환
    _size1 = data.shape
    if log:
        for i in range(_size1[0]):
            data['L3061_tg'].iloc[i] = np.log(data['L3061_tg'].iloc[i])

    # PHx_DL_MED : 이상지질혈증 투약 0&missing -> 미복용
    data['PHx_DL_MED'] = np.where(pd.notnull(data['PHx_DL_MED']) == True, data['PHx_DL_MED'], 0)

    # exercise 변수 생성
    data['IEX_Hx_DPW'] = np.where(pd.notnull(data['IEX_Hx_DPW']) == True, data['IEX_Hx_DPW'], 0)
    data['MEX_Hx_DPW'] = np.where(pd.notnull(data['MEX_Hx_DPW']) == True, data['MEX_Hx_DPW'], 0)
    data['exercise'] = data['IEX_Hx_DPW'] + data['MEX_Hx_DPW']
    data = data.loc[data.exercise <= 14, :]

    # smoking
    data['SMK_Hx_0'] = np.where(pd.notnull(data['SMK_Hx_0']) == True, data['SMK_Hx_0'], 0)
    data['SMK_Hx_1'] = np.where(pd.notnull(data['SMK_Hx_1']) == True, data['SMK_Hx_1'], 0)
    data['SMK_Hx_2'] = np.where(pd.notnull(data['SMK_Hx_2']) == True, data['SMK_Hx_2'], 0)

    data['smoking'] = data['SMK_Hx_0'] + data['SMK_Hx_1'] + data['SMK_Hx_2']
    data = data.loc[data.smoking == 1, :]

    # drinking
    data['DR_Hx_0'] = np.where(pd.notnull(data['DR_Hx_0']) == True, data['DR_Hx_0'], 0)
    data['DR_Hx_1'] = np.where(pd.notnull(data['DR_Hx_1']) == True, data['DR_Hx_1'], 0)
    data['DR_Hx_2'] = np.where(pd.notnull(data['DR_Hx_2']) == True, data['DR_Hx_2'], 0)

    data['drinking'] = data['DR_Hx_0'] + data['DR_Hx_1'] + data['DR_Hx_2']
    data = data.loc[data.drinking == 1, :]

    # 수면시간(SL_BdTH)
    data = data.loc[data.SL_BdTH <= 24, :]

    data = data[['RAND_KEY', 'AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight',
                 'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp', 'HPC00006_pulse',
                 'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast',
                 'L3015_alt', 'SEX_TP_CD', 'BMI', 'WtH',
                 'PHx_DL_MED', 'smoking', 'drinking', 'exercise', 'SL_BdTH',
                 'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    data = data.drop('RAND_KEY', axis=1).dropna(axis=0)
    xdata = data[data.columns.difference(['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl'])]
    ydata = data[['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    return data, xdata, ydata



def DATA_PREPROCESS_SH_origin():
    """
        김시현 선생님이 만든 원본
    """
    data = pd.read_csv('AI_DATA_SORT.csv')
    size = data.shape

    data = data.dropna(subset=['L3008_cholesterol', 'L3061_tg', 'L3062_hdl'])

    # 0 for Female, 1 for Male
    data['SEX_TP_CD'] = data.apply(lambda row: 0 if row['SEX_TP_CD'] == 'F' else 1, axis=1)

    # Control date type
    data['ORD_DT'] = data.RAND_KEY.str[-10:]
    data['ORD_DT'] = data['ORD_DT'].astype('datetime64[ns]')
    data['PT_BRDY_DT'] = data['PT_BRDY_DT'].astype('datetime64[ns]')
    data['AGE'] = (data['ORD_DT'] - data['PT_BRDY_DT']).dt.days / 365.25

    # age >= 20
    data = data.loc[data.AGE >= 20, :]
    # Remove useless variables
    data = data[data.columns.difference(['RAND', 'PT_BRDY_DT', 'ORD_DT'])]

    # Remove rows with string data
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43]].astype(float)
        except ValueError as e:
            remove_row.append(i)

    data = data.drop(data.index[remove_row])
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43]:
        name = data.columns[i]
        data = data.astype({name: 'float'})

    data['BMI'] = data['HPC00002_weight'] / (data['HPC00001_height'] / 100) ** 2
    data['WtH'] = data['HPC00007_wc'] / data['HPC00001_height']

    # ldl imputation
    data['L3068_ldl'] = np.where(pd.notnull(data['L3068_ldl']) == True, data['L3068_ldl'],
                                 data['L3008_cholesterol'] - ((data['L3061_tg'] / 5) + data['L3062_hdl']))

    # TG log 변환
    _size1 = data.shape
    for i in range(_size1[0]):
        data['L3061_tg'].iloc[i] = np.log(data['L3061_tg'].iloc[i])

    # PHx_DL_MED : 이상지질혈증 투약 0&missing -> 미복용
    data['PHx_DL_MED'] = np.where(pd.notnull(data['PHx_DL_MED']) == True, data['PHx_DL_MED'], 0)

    # exercise 변수 생성
    data['IEX_Hx_DPW'] = np.where(data['IEX_Hx_DPW'] == 1, data['IEX_Hx_DPW'], 0)
    data['MEX_Hx_DPW'] = np.where(pd.notnull(data['MEX_Hx_DPW']) == True, data['MEX_Hx_DPW'], 0)
    data['exercise'] = data['IEX_Hx_DPW'] + data['MEX_Hx_DPW']
    data = data.loc[data.exercise <= 14, :]

    # smoking
    data['SMK_Hx_0'] = np.where(pd.notnull(data['SMK_Hx_0']) == True, data['SMK_Hx_0'], 0)
    data['SMK_Hx_1'] = np.where(pd.notnull(data['SMK_Hx_1']) == True, data['SMK_Hx_1'], 0)
    data['SMK_Hx_2'] = np.where(pd.notnull(data['SMK_Hx_2']) == True, data['SMK_Hx_2'], 0)

    data['smoking'] = data['SMK_Hx_0'] + data['SMK_Hx_1'] + data['SMK_Hx_2']
    data = data.loc[data.smoking == 1, :]

    # drinking
    data['DR_Hx_0'] = np.where(pd.notnull(data['DR_Hx_0']) == True, data['DR_Hx_0'], 0)
    data['DR_Hx_1'] = np.where(pd.notnull(data['DR_Hx_1']) == True, data['DR_Hx_1'], 0)
    data['DR_Hx_2'] = np.where(pd.notnull(data['DR_Hx_2']) == True, data['DR_Hx_2'], 0)

    data['drinking'] = data['DR_Hx_0'] + data['DR_Hx_1'] + data['DR_Hx_2']
    data = data.loc[data.drinking == 1, :]

    # 수면시간(SL_BdTH)
    data = data.loc[data.SL_BdTH <= 24, :]

    data = data[['RAND_KEY', 'AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight',
                 'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp', 'HPC00006_pulse',
                 'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast',
                 'L3015_alt', 'SEX_TP_CD', 'BMI', 'WtH',
                 'PHx_DL_MED', 'smoking', 'drinking', 'exercise', 'SL_BdTH',
                 'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    data_no_na = data.drop('RAND_KEY', axis=1).dropna(axis=0)

    return data, data_no_na


def DATA_PREPROCESS_SH_v2():
    data = pd.read_csv('AI_DATA_SORT.csv')
    size = data.shape

    data = data.dropna(subset=['L3008_cholesterol', 'L3061_tg', 'L3062_hdl'])

    # 0 for Female, 1 for Male
    data['SEX_TP_CD'] = data.apply(lambda row: 0 if row['SEX_TP_CD'] == 'F' else 1, axis=1)

    # Control date type
    data['ORD_DT'] = data.RAND_KEY.str[-10:]
    data['ORD_DT'] = data['ORD_DT'].astype('datetime64[ns]')
    data['PT_BRDY_DT'] = data['PT_BRDY_DT'].astype('datetime64[ns]')
    data['AGE'] = (data['ORD_DT'] - data['PT_BRDY_DT']).dt.days / 365.25

    # age >= 20
    data = data.loc[data.AGE >= 20, :]

    # Remove rows with string data
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[
                i, [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46]].astype(float)
        except ValueError as e:
            remove_row.append(i)

    data = data.drop(data.index[remove_row])
    for i in [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46]:
        name = data.columns[i]
        data = data.astype({name: 'float'})

    data['BMI'] = data['HPC00002_weight'] / (data['HPC00001_height'] / 100) ** 2
    data['WtH'] = data['HPC00007_wc'] / data['HPC00001_height']

    # ldl imputation
    data['L3068_ldl'] = np.where(pd.notnull(data['L3068_ldl']) == True, data['L3068_ldl'],
                                 data['L3008_cholesterol'] - ((data['L3061_tg'] / 5) + data['L3062_hdl']))

    # TG log 변환
    _size1 = data.shape
    data['L3061_tg'] = np.log(data['L3061_tg'])

    data_2 = data.copy()

    # PHx_DL_MED : 이상지질혈증 투약 0&missing -> 미복용
    data['PHx_DL_MED'] = np.where(pd.notnull(data['PHx_DL_MED']) == True, data['PHx_DL_MED'], 0)

    # exercise 변수 생성
    data['IEX_Hx_DPW'] = np.where(pd.notnull(data['IEX_Hx_DPW']) == True, data['IEX_Hx_DPW'], 0)
    data['MEX_Hx_DPW'] = np.where(pd.notnull(data['MEX_Hx_DPW']) == True, data['MEX_Hx_DPW'], 0)
    data['exercise'] = data['IEX_Hx_DPW'] + data['MEX_Hx_DPW']
    data = data.loc[data.exercise <= 14, :]

    # smoking
    data['SMK_Hx_0'] = np.where(pd.notnull(data['SMK_Hx_0']) == True, data['SMK_Hx_0'], 0)
    data['SMK_Hx_1'] = np.where(pd.notnull(data['SMK_Hx_1']) == True, data['SMK_Hx_1'], 0)
    data['SMK_Hx_2'] = np.where(pd.notnull(data['SMK_Hx_2']) == True, data['SMK_Hx_2'], 0)

    data['smoking'] = data['SMK_Hx_0'] + data['SMK_Hx_1'] + data['SMK_Hx_2']
    data = data.loc[data.smoking == 1, :]
    data['smoking'] = np.where(data['SMK_Hx_0'] == 1, 0,
                               np.where(data['SMK_Hx_1'] == 1, 1, 2))

    # drinking
    data['DR_Hx_0'] = np.where(pd.notnull(data['DR_Hx_0']) == True, data['DR_Hx_0'], 0)
    data['DR_Hx_1'] = np.where(pd.notnull(data['DR_Hx_1']) == True, data['DR_Hx_1'], 0)
    data['DR_Hx_2'] = np.where(pd.notnull(data['DR_Hx_2']) == True, data['DR_Hx_2'], 0)

    data['drinking'] = data['DR_Hx_0'] + data['DR_Hx_1'] + data['DR_Hx_2']
    data = data.loc[data.drinking == 1, :]
    data['drinking'] = np.where(data['DR_Hx_0'] == 1, 0,
                                np.where(data['DR_Hx_1'] == 1, 1, 2))

    # 수면시간(SL_BdTH)
    data = data.loc[data.SL_BdTH <= 24, :]

    data = data[['RAND_KEY', 'AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight',
                 'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp', 'HPC00006_pulse',
                 'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast',
                 'L3015_alt', 'SEX_TP_CD', 'BMI', 'WtH',
                 'PHx_DL_MED', 'smoking', 'drinking', 'exercise', 'SL_BdTH',
                 'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    # last lipid profile - from new dataset
    chol = pd.read_csv('supreme_chol.csv', encoding='cp949').loc[:, ['RAND', 'RAND_KEY', '검사시행일', '검사결과-수치값']];
    chol['name'] = 'L3008_cholesterol'
    tg = pd.read_csv('supreme_tg.csv', encoding='cp949').loc[:, ['RAND', 'RAND_KEY', '검사시행일', '검사결과-수치값']];
    tg['name'] = 'L3061_tg'
    hdl = pd.read_csv('supreme_hdl.csv', encoding='cp949').loc[:, ['RAND', 'RAND_KEY', '검사시행일', '검사결과-수치값']];
    hdl['name'] = 'L3062_hdl'
    ldl = pd.read_csv('supreme_ldl.csv', encoding='cp949').loc[:, ['RAND', 'RAND_KEY', '검사시행일', '검사결과-수치값']];
    ldl['name'] = 'L3068_ldl'

    last_lp = pd.concat([chol, tg, hdl, ldl], axis=0)
    last_lp.columns = ['RAND', 'RAND_KEY', 'ORD_DT', 'value', 'name']
    last_lp = last_lp.drop_duplicates(['RAND_KEY', 'ORD_DT', 'name'], keep='first')
    last_lp = last_lp.set_index(['RAND', 'RAND_KEY', 'ORD_DT']).pivot(columns='name').reset_index().dropna(axis=0)
    last_lp.columns = ['RAND', 'RAND_KEY', 'ORD_DT', 'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']

    cnt = last_lp.groupby('RAND').count().ORD_DT
    data_3 = last_lp.loc[last_lp.RAND.isin(list(cnt[cnt >= 2].index))]
    data_3 = data_3.sort_values(by=['RAND', 'RAND_KEY', 'ORD_DT'])
    shifted = data_3.groupby('RAND').shift(1)

    data_3 = data_3.join(shifted.rename(columns=lambda x: x + '_shifted')).loc[:,
             ['RAND_KEY', 'ORD_DT', 'ORD_DT_shifted', 'L3008_cholesterol_shifted', 'L3061_tg_shifted',
              'L3062_hdl_shifted', 'L3068_ldl_shifted']].dropna(axis=0)
    data_3['ORD_DT'] = data_3['ORD_DT'].astype('datetime64[ns]')
    data_3['ORD_DT_shifted'] = data_3['ORD_DT_shifted'].astype('datetime64[ns]')
    data_3['interval'] = data_3.ORD_DT - data_3.ORD_DT_shifted
    data_3['interval'] = data_3['interval'].dt.days

    data_3 = data_3.loc[:,
             ['RAND_KEY', 'L3008_cholesterol_shifted', 'L3061_tg_shifted', 'L3062_hdl_shifted', 'L3068_ldl_shifted',
              'interval']]

    # merge shifted var. and data
    data_4 = pd.merge(data, data_3, on='RAND_KEY').drop('RAND_KEY', axis=1).dropna(axis=0)
    data_4 = data_4[
        ['AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight', 'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp',
         'HPC00006_pulse',
         'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast', 'L3015_alt', 'SEX_TP_CD', 'BMI',
         'WtH', 'PHx_DL_MED',
         'smoking', 'drinking', 'exercise', 'SL_BdTH',
         'L3008_cholesterol_shifted', 'L3061_tg_shifted', 'L3062_hdl_shifted', 'L3068_ldl_shifted', 'interval',
         'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    # last lipid profile - from original dataset
    cnt2 = data_2.groupby('RAND').count().ORD_DT
    data_2 = data_2.loc[data_2.RAND.isin(list(cnt2[cnt2 >= 2].index))]
    shifted = data_2.groupby('RAND').shift(1)

    data_2 = data_2.join(shifted.rename(columns=lambda x: x + '_shifted')).loc[:,
             ['RAND_KEY', 'ORD_DT', 'ORD_DT_shifted', 'L3008_cholesterol_shifted', 'L3061_tg_shifted',
              'L3062_hdl_shifted', 'L3068_ldl_shifted']].dropna(axis=0)
    data_2['ORD_DT'] = data_2['ORD_DT'].astype('datetime64[ns]')
    data_2['ORD_DT_shifted'] = data_2['ORD_DT_shifted'].astype('datetime64[ns]')
    data_2['interval'] = data_2.ORD_DT - data_2.ORD_DT_shifted
    data_2['interval'] = data_2['interval'].dt.days

    data_2 = data_2.loc[:,
             ['RAND_KEY', 'L3008_cholesterol_shifted', 'L3061_tg_shifted', 'L3062_hdl_shifted', 'L3068_ldl_shifted',
              'interval']]

    # merge shifted var. and data
    data_5 = pd.merge(data, data_2, on='RAND_KEY').drop('RAND_KEY', axis=1).dropna(axis=0)
    data_5 = data_5[
        ['AGE', 'FATTY_DEG', 'HPC00001_height', 'HPC00002_weight', 'HPC00003_fat_ratio', 'HPC00004_sbp', 'HPC00005_dbp',
         'HPC00006_pulse',
         'HPC00007_wc', 'HPCL7606_tsh', 'HPCL7624_ft4', 'L3005_glucose', 'L3014_ast', 'L3015_alt', 'SEX_TP_CD', 'BMI',
         'WtH', 'PHx_DL_MED',
         'smoking', 'drinking', 'exercise', 'SL_BdTH',
         'L3008_cholesterol_shifted', 'L3061_tg_shifted', 'L3062_hdl_shifted', 'L3068_ldl_shifted', 'interval',
         'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    # concat data_4, data_5
    data_6 = pd.concat([data_4, data_5], axis=0)
    data_6.L3061_tg_shifted = np.log(data_6.L3061_tg_shifted)
    data_6 = data_6.drop_duplicates()

    data = data.drop('RAND_KEY', axis=1)

    # categorize
    data_7 = data_6.copy()

    data_7['L3008_cholesterol'] = np.where(data_7['L3008_cholesterol'] < 200, '적정',
                                           np.where(data_7['L3008_cholesterol'] < 240, '경계', '높음'))
    data_7['L3061_tg'] = np.where(data_7['L3061_tg'] < np.log(150), '적정',
                                  np.where(data_7['L3061_tg'] < np.log(200), '경계',
                                           np.where(data_7['L3061_tg'] < np.log(500), '높음', '매우 높음')))
    data_7['L3062_hdl'] = np.where(data_7['L3062_hdl'] < 40, '낮음',
                                   np.where(data_7['L3062_hdl'] < 60, '적정', '높음'))
    data_7['L3068_ldl'] = np.where(data_7['L3068_ldl'] < 100, '적정',
                                   np.where(data_7['L3068_ldl'] < 130, '정상',
                                            np.where(data_7['L3068_ldl'] < 160, '경계',
                                                     np.where(data_7['L3068_ldl'] < 190, '높음', '매우높음'))))

    # normal / abnormal
    data_8 = data_6.copy()

    data_8['L3008_cholesterol'] = np.where(data_8['L3008_cholesterol'] < 200, '정상', '비정상')
    data_8['L3061_tg'] = np.where(data_8['L3061_tg'] < np.log(150), '정상', '비정상')
    data_8['L3062_hdl'] = np.where(data_8['L3062_hdl'] < 40, '비정상',
                                   np.where(data_8['L3062_hdl'] < 60, '정상', '비정상'))
    data_8['L3068_ldl'] = np.where(data_8['L3068_ldl'] < 130, '정상', '비정상')


    return data, data_6, data_7, data_8





def FINAL_DATA():
    df_sort, df_sort_1 = DATA_PREPROCESS_SH_origin()

    data = pd.read_csv('AI_DATA_SORT.csv')

    # ORD_DT NA imputation
    data['ORD_DT'] = data.RAND_KEY.str[-10:]

    data = data[['RAND', 'RAND_KEY', 'ORD_DT', 'L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    data = data.dropna(subset=['L3008_cholesterol', 'L3061_tg', 'L3062_hdl'])

    # as.float
    datasize = data.shape
    remove_row = []
    for i in range(datasize[0]):
        try:
            tmp = data.iloc[i, [3, 4, 5, 6]].astype(float)
        except ValueError as e:
            remove_row.append(i)

    data = data.drop(data.index[remove_row])

    data.L3008_cholesterol = data.L3008_cholesterol.astype(float)
    data.L3061_tg = data.L3061_tg.astype(float)
    data.L3062_hdl = data.L3062_hdl.astype(float)
    data.L3068_ldl = data.L3068_ldl.astype(float)

    # ldl imputation
    data['L3068_ldl'] = np.where(pd.notnull(data['L3068_ldl']) == True, data['L3068_ldl'],
                                 data['L3008_cholesterol'] - ((data['L3061_tg'] / 5) + data['L3062_hdl']))

    lab_count = data.groupby('RAND').count().ORD_DT

    # ppl who has last lipid profile values
    data2 = data.loc[data.RAND.isin(list(lab_count[lab_count >= 2].index))]
    # 이전 값
    shifted = data2.groupby('RAND').shift(1)

    # merge shifted and data
    data2 = data2.join(shifted.rename(columns=lambda x: x + '_shifted')).dropna(axis=0).loc[:,
            ['RAND_KEY', 'L3008_cholesterol_shifted', 'L3061_tg_shifted', 'L3062_hdl_shifted', 'L3068_ldl_shifted']]

    # merge shifted var. and data
    df_sort_2 = pd.merge(df_sort, data2, on='RAND_KEY').drop('RAND_KEY', axis=1).dropna(axis=0)
    data = df_sort_2.copy()
    xdata = df_sort_2[df_sort_2.columns.difference(['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl'])]
    ydata = df_sort_2[['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    return data, xdata, ydata



def NORMALIZATION(data, mode):
    if mode == 'standard':
        scaler = StandardScaler()
    if mode == 'minmax':
        scaler = MinMaxScaler()

    normalized = scaler.fit_transform(data)

    return normalized


def CATEGORIZATION(ydata):
    TCH = []
    TG = []
    HDL = []
    LDL = []

    for i in ydata:
        if i[-4] < 200:
            TCH.append(0)
        elif i[-4] < 240:
            TCH.append(1)
        else:
            TCH.append(2)

        if i[-3] < np.log(150):
            TG.append(0)
        elif i[-3] < np.log(200):
            TG.append(1)
        else:
            TG.append(2)

        if i[-2] < 40:
            HDL.append(0)
        elif i[-2] < 60:
            HDL.append(1)
        else:
            HDL.append(2)

        if i[-1] < 100:
            LDL.append(0)
        elif i[-1] < 130:
            LDL.append(1)
        elif i[-1] < 160:
            LDL.append(2)
        elif i[-1] < 190:
            LDL.append(3)
        else:
            LDL.append(4)

    return np.transpose([TCH, TG, HDL, LDL])


def CATEGORIZATION_v2(ydata):
    TCH = []
    TG = []
    HDL = []
    LDL = []

    for i in ydata:
        if i[-4] < 200:
            TCH.append(0)
        else:
            TCH.append(1)

        if i[-3] < np.log(150):
            TG.append(0)
        else:
            TG.append(1)

        if i[-2] < 60:
            HDL.append(0)
        else:
            HDL.append(1)

        if i[-1] < 130:
            LDL.append(0)
        else:
            LDL.append(1)

    return np.transpose([TCH, TG, HDL, LDL])


def REMOVE_AMBIGUOUS(data, y_value):
    if y_value == -4:
        removed = np.array([num for num, i in enumerate(data) if 240 <= i[-1] or i[-1] < 180])
    elif y_value == -3:
        removed = np.array([num for num, i in enumerate(data) if np.log(200) <= i[-1] or i[-1] < np.log(120)])
    elif y_value == -2:
        removed = np.array([num for num, i in enumerate(data) if 80 <= i[-1] or i[-1] < 40])
    elif y_value == -1:
        removed = np.array([num for num, i in enumerate(data) if 160 <= i[-1] or i[-1] < 100])

    return removed


def xysplit(data):
    xdata = data[data.columns.difference(['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl'])]
    ydata = data[['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']]

    return xdata, ydata



















