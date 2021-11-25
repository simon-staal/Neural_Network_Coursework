import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute
from sklearn.utils.estimator_checks import check_estimator
import part2_house_value_regression as lib

def TestLoadScore():
    data = pd.read_csv("housing.csv")

    output_label = "median_house_value"
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = lib.load_regressor()

    print(regressor.score(x, y))

def TestHyperParams():
    data = pd.read_csv("housing.csv")

    output_label = "median_house_value"
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    params = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        'nb_epoch': [500, 1000, 2000],
        'neurons': [[9, 5, 1], [15, 8, 1], [8, 8, 8, 1]]
    }

    res = lib.RegressorHyperParameterSearch(x_train, y_train, params)

    print(res)
    

def TestProximity():
    data = pd.read_csv("housing.csv")

    output_label = "median_house_value"
    x_train = data.loc[:, data.columns != output_label]

    ocean_proximity = x_train['ocean_proximity'].to_numpy()
    print(ocean_proximity)

def TestPreproc():
    data = pd.read_csv("housing.csv")

    output_label = "median_house_value"
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train = x
    #print(x_train[:][:3])
    x_train['ocean_proximity'][2] = np.nan
    #print(x_train[:][:3])
    #print(x_train['ocean_proximity'].mode()[0])

    regressor = lib.Regressor(x_train, nb_epoch = 10)
    #print(x_train[:][:3])

    data_sub = x_train[:5].copy()
    data_sub['ocean_proximity'][3] = np.nan
    #print(data_sub)
    regressor._preprocessor(data_sub)

    test = 8
    dev = 1
    train = 1

    x_size = len(x.index)
    fold_size = x_size // (test + dev + train)

    permutation = torch.randperm(x_size)
    test_split = permutation[:fold_size * test]
    dev_split = permutation[fold_size * test:fold_size * (test + dev)]
    train_split = permutation[fold_size * (test + dev):]

    x_train = x.iloc[train_split]
    y_train = y.iloc[train_split]

    x_dev = x.iloc[dev_split]
    y_dev = y.iloc[dev_split]

    regressor.fit(x_train, y_train)

if __name__ == "__main__":
    TestHyperParams()