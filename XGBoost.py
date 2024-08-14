import numpy as np
import progressbar
import pandas as pd
from my_dataset import *
from datetime import *
import logging
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

np.random.seed(10)

def main(max_depth= 5, n_estimators=5):
    X_train, X_test, y_train, y_test = gen_datasets()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置XGBoost的参数
    params = {
        'objective': 'reg:squarederror',  # 目标函数
        'max_depth': max_depth,  # 树的最大深度
        'eta': 0.01,  # 学习率
        'subsample': 1,  # 子采样比例
        'colsample_bytree': 0.8,  # 每棵树的特征子采样比例
        'seed': 42  # 随机种子
    }

    num_round = n_estimators  # 迭代次数
    bst = xgb.train(params, dtrain, num_round)
    y_pred = bst.predict(dtest)


    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f'Results: MSE:{mse_test:.4f}')
    logger.info(f'Results: MAE:{mae_test:.4f}')
    logger.info(f'Results: R2:{r2_test:.4f}')
    logger.info(f'Results: RMSE:{rmse:.4f}')

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    max_depth = 5
    n_estimators = 5
    logging.basicConfig(filename=f'xgb_log/XGBoost_{max_depth}_{n_estimators}.log', level=logging.INFO)
    logger.info(f'Start...')
    logger.info(f'max_depth: {max_depth}')
    logger.info(f'n_estimators: {n_estimators}')
    start = datetime.now()
    main(max_depth, n_estimators)
    end = datetime.now()
    logger.info('Time: {}'.format(end - start))

    print(end - start)
    logger.info('End')


