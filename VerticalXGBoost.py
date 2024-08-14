import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
from SSCalculation import *
from Tree import *
import math
import time
from my_dataset import *
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

np.random.seed(10)
clientNum = 4

class LeastSquareLoss:
    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) # Mind the dimension

class VerticalXGBoostClassifier:

    def __init__(self, rank, lossfunc, splitclass, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, max_depth=3):
        if lossfunc == 'LogLoss':
            self.loss = LogLoss()
        else:
            self.loss = LeastSquareLoss()
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self.n_estimators = n_estimators  # Number of trees
        self.max_depth = max_depth  # Maximum depth for tree
        self.rank = rank
        self.trees = []
        self.splitclass = splitclass
        for _ in range(n_estimators):
            tree = VerticalXGBoostTree(rank=self.rank,
                                       lossfunc=self.loss,
                                       splitclass=self.splitclass,
                                       _lambda=self._lambda,
                                        _gamma=self._gamma,
                                       _epsilon=self._epsilon,
                                       _maxdepth=self.max_depth,
                                       clientNum=clientNum)
            self.trees.append(tree)

    def getQuantile(self, colidx):
        split_list = []
        if self.rank != 0: # For client nodes
            data = self.data.copy()
            idx = np.argsort(data[:, colidx], axis=0)
            data = data[idx]
            value_list = sorted(list(set(list(data[:, colidx]))))  # Record all the different value
            hess = np.ones_like(data[:, colidx])
            data = np.concatenate((data, hess.reshape(-1, 1)), axis=1)
            sum_hess = np.sum(hess)
            last = value_list[0]
            i = 1
            if len(value_list) == 1: # For those who has only one value, do such process.
                last_cursor = last
            else:
                last_cursor = value_list[1]
            split_list.append((-np.inf, value_list[0]))
            # if len(value_list) == 15000:
            #     print(self.rank, colidx)
            #     print(value_list)
            while i < len(value_list):
                cursor = value_list[i]
                small_hess = np.sum(data[:, -1][data[:, colidx] <= last]) / sum_hess
                big_hess = np.sum(data[:, -1][data[:, colidx] <= cursor]) / sum_hess
                # print(colidx, self.rank, np.abs(big_hess - small_hess), last, cursor)
                if np.abs(big_hess - small_hess) < self._epsilon:
                    last_cursor = cursor
                else:
                    judge = value_list.index(cursor) - value_list.index(last)
                    if judge == 1: # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                        split_list.append((last, cursor))
                        last = cursor
                    else: # Move forward and record the last.
                        split_list.append((last, last_cursor))
                        last = last_cursor
                        last_cursor = cursor
                i += 1
            if split_list[-1][1] != value_list[-1]:
                split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
            split_list = np.array(split_list)
        return split_list

    def getAllQuantile(self): # Global quantile, must be calculated before tree building, avoiding recursion.
        self_maxlen = 0
        if self.rank != 0:
            dict = {i:self.getQuantile(i) for i in range(self.data.shape[1])} # record all the split
            self_maxlen = max([len(dict[i]) for i in dict.keys()])
        else:
            dict = {}

        recv_maxlen = comm.gather(self_maxlen, root=1)
        maxlen = None
        if self.rank == 1:
            maxlen = max(recv_maxlen)

        self.maxSplitNum = comm.bcast(maxlen, root=1)
        # print('MaxSplitNum: ', self.maxSplitNum)
        self.quantile = dict

    def fit(self, X, y):
        data_num = X.shape[0]
        y = np.reshape(y, (data_num, 1))
        y_pred = np.zeros(np.shape(y))
        self.data = X.copy()
        self.getAllQuantile()
        for i in range(self.n_estimators):
            # print('In classifier fit, rank: ', self.rank)
            tree = self.trees[i]
            tree.data, tree.maxSplitNum, tree.quantile = self.data, self.maxSplitNum, self.quantile
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(y_and_pred, i)
            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = tree.predict(X)
            if self.rank == 1:
                # print('test')
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred

    def predict(self, X):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main(max_depth=5,
         n_estimators=5):

    X_train, X_test, y_train, y_test = gen_datasets()

    X_train_A = X_train[:, :2]
    X_train_B = X_train[:, 2:4]
    X_train_C = X_train[:, 4:7]
    X_train_D = X_train[:, 7:]

    X_test_A = X_test[:, :2]
    X_test_B = X_test[:, 2:4]
    X_test_C = X_test[:, 4:7]
    X_test_D = X_test[:, 7:]

    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank,
                                      lossfunc='reg',
                                      splitclass=splitclass,
                                      max_depth=max_depth,
                                      n_estimators=n_estimators, _epsilon=0.1)
    logger.info('Started')
    logger.info('max_depth:', max_depth)
    logger.info('n_estimators:', n_estimators)
    start = datetime.now()
    if rank == 1:
        logger.info('Rank: ', rank)
        model.fit(X_train_A, y_train)
        end = datetime.now()
        print('In fitting 1: ', end - start)
        logger.info('In fitting 1: ', end - start)
        time = end - start
        for i in range(clientNum + 1):
            if i == 1:
                pass
            else:
                time += comm.recv(source=i)
        print(time / (clientNum + 1))
        final_time = time / (clientNum + 1)
        print('end 1')
        print(final_time)
        logger.info('end 1...')
        logger.info(f'training time:{final_time}')
    elif rank == 2:
        logger.info('Rank: ', rank)
        model.fit(X_train_B, np.zeros_like(y_train))
        end = datetime.now()
        logger.info('In fitting 2: ', end - start)
        logger.info('end 2')
        comm.send(end - start, dest=1)
        print('In fitting 2: ', end - start)
        print('end 2')

    elif rank == 3:
        logger.info('Rank: ', rank)
        model.fit(X_train_C, np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 3: ', end - start)
        logger.info('In fitting 3: ', end - start)
        logger.info('end 3')
        comm.send(end - start, dest=1)
        print('end 3')

    elif rank == 4:
        logger.info('Rank: ', rank)
        model.fit(X_train_D, np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 4: ', end - start)
        logger.info('In fitting 4: ', end - start)
        logger.info('end 4')
        comm.send(end - start, dest=1)
        print('end 4')

    else:
        logger.info('Rank: ', rank)
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        end = datetime.now()
        print('In fitting 0: ', end - start)
        logger.info('In fitting 0: ', end - start)
        logger.info('end 0')
        comm.send(end - start, dest=1)
        print('end 0')

    # print("rank == ", rank)

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        logger.info(f'Results: MSE:{mse_test:.4f}')
        logger.info(f'Results: MAE:{mae_test:.4f}')
        logger.info(f'Results: R2:{r2_test:.4f}')
        logger.info(f'Results: RMSR:{rmse:.4f}')

        print("\nTesting Metrics:")
        print(f"Mean Squared Error: {mse_test:.4f}")
        print(f"Mean Absolute Error: {mae_test:.4f}")
        print(f"R-squared: {r2_test:.4f}")
        print(f"RMSR: {rmse:.4f}")

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    max_depth = 3
    n_estimators = 3
    logging.basicConfig(filename=f'log/FedXGBoost_{max_depth}_{n_estimators}.log', level=logging.INFO)

    main(max_depth, n_estimators)

