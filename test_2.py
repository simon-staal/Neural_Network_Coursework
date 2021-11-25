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

    regressor = lib.Regressor(x_train, nb_epoch = 100)
    regressor.fit(x_train, y_train)
    params = {
        'learning_rate': [0.001, 0.005],
        'nb_epoch': [100, 500]
    }

    res = lib.RegressorHyperParameterSearch(regressor, x_train, y_train, params)

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
    x_train = data.loc[:, data.columns != output_label]
    print(x_train[:][:3])
    x_train['ocean_proximity'][2] = np.nan
    print(x_train[:][:3])
    print(x_train['ocean_proximity'].mode()[0])

    regressor = lib.Regressor(x_train, nb_epoch = 10)
    print(x_train[:][:3])

    data_sub = x_train[:5].copy()
    data_sub['ocean_proximity'][3] = np.nan
    print(data_sub)
    print(regressor._preprocessor(data_sub))

if __name__ == "__main__":
    TestHyperParams()