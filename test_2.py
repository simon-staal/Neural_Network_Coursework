import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute
import part2_house_value_regression as lib

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
    TestPreproc()