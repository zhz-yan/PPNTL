import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# print(pd.__version__)

def gen_datasets():

    y = pd.read_csv('data/basic.csv', header=0)
    x = pd.read_csv('data/details.csv', header=0)

    df = x[['数据日期', '台区编号', '上网电量占比',
             '末端电量占比', '首末端压降', '功率因数',
             '负载率', '负荷形状系数', '三相不平衡度',
             '供电半径', '网架结构', '用户总数', '数据情况']]

    df_y = y[['数据日期', '台区编号', '供电量', '售电量', '线损率', '理论线损率']]
    df['数据日期'] = pd.to_datetime(df['数据日期'], format='%Y-%m-%d')
    df_y['数据日期'] = pd.to_datetime(df_y['数据日期'], format='%Y-%m-%d')
    df = pd.merge(df, df_y, how='left', on=['台区编号', '数据日期'])
    df = df.drop(df[df['数据情况'] != '数据正常'].index)

    # 剔除有重复数据 并 按日期排序
    df = df.sort_values(by=['台区编号', '数据日期'])
    df = df[df.duplicated() == False]
    df['数据日期'] = pd.to_datetime(df['数据日期'], format='%Y/%m/%d')
    df = df.sort_values(by=['台区编号', '数据日期'])

    ID_NET_MAP = {1: '电缆', 2: '架空绝缘', 3: '架空裸导', 4: '混合'}
    NET_ID_MAP = dict((a, i) for i, a in ID_NET_MAP.items())
    df['网架结构'] = df['网架结构'].map(NET_ID_MAP)
    df['网架结构'] = df['网架结构'].values
    # 提取时间 2021-04-13 至 2021-04-13

    open_day_1 = '2021-04-01'
    close_day_1 = '2021-04-01'

    open_day_2 = '2021-04-10'
    close_day_2 = '2021-04-10'

    df_1 = df['数据日期'] >= open_day_1
    df_2 = df['数据日期'] <= close_day_1

    df_3 = df['数据日期'] >= open_day_2
    df_4 = df['数据日期'] <= close_day_2

    df_train = df[df_1 & df_2]
    df_train = df_train.apply(pd.to_numeric, errors='ignore')

    df_test = df[df_3 & df_4]
    df_test = df_test.apply(pd.to_numeric, errors='ignore')

    df_train = df_train[df_train["理论线损率"] > 0]
    df_test = df_test[df_test["理论线损率"] > 0]
    # df = df[df["线损率"] < 100]

    df_train.set_index(['台区编号'], inplace=True)
    df_train.drop(['数据日期', '数据情况', '售电量', '线损率'], inplace=True, axis=1)

    df_test.set_index(['台区编号'], inplace=True)
    df_test.drop(['数据日期', '数据情况', '售电量', '线损率'], inplace=True, axis=1)

    X_train = pd.DataFrame(df_train[['上网电量占比', '末端电量占比', '首末端压降', '功率因数',
                                     '负载率', '负荷形状系数', '三相不平衡度', '供电半径', '网架结构',
                                     '用户总数', '供电量']], dtype='float')
    # X = pd.to_numeric(X, errors='ignore')
    y_train = df_train['理论线损率']

    X_test = pd.DataFrame(df_test[['上网电量占比', '末端电量占比', '首末端压降', '功率因数',
                                   '负载率', '负荷形状系数', '三相不平衡度', '供电半径', '网架结构',
                                   '用户总数', '供电量']], dtype='float')
    # X = pd.to_numeric(X, errors='ignore')
    y_test = df_test['理论线损率']

    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = gen_datasets()