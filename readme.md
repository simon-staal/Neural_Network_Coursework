Artificial Neural Networks ML Coursework
===========================
This repository contains the python implementation of a neural network mini-library and a complete regression neural netwrok trained to estimate housing prices produced as part of COMP70050 - Introduction to Machine Learning (Autmumn 2021). Please find a brief guide to the repository below:

Contributors
------------
- Simon Staal (sts219)
- Petra Ratkai (petraratkai)
- Thomas Loureiro van Issum (tl319)
- David Cormier (DavidMael)

Overview
--------
1. Neural network mini-library
The neural network mini-library can be found [**here**](part1_nn_lib.py). When running this file with the `python3 part1_nn_lib.py` command, a simple test on the library gets executed.

2. Trained neural net
The implementation of the trained neural network can be found [**here**](part2_house_value_regression.py). This file can be run using the `python3 part2_house_value_regression.py` command. This file splits the data into training, validation and test sets, trains the neural network using the training and validation sets and evaluates the result suing the test set. The regressor in this file is initialised with the best parameters we have obtained.

[**test2_py**](test2_py/)
-----
This file can be run using the `python3 test2_py` command. This file can be used to test the hyperparameter tuning of the neural network. The program creates a search space of different parameters to be tested and then calls the `RegressorHyperParameterSearch(x, y, params)` function that tunes the hyperparameters of the model using grid search cross validation.

[**iris.dat**](iris.dat)
-----
This file contains the Iris dataset which was used to test the neural network mini-library.

[**houseing.csv**](housing.csv)
-----
This file contains the raw data our neural network was trained on.

[**part2_model.pickle**](part2_model_pickle)
-----
The pickle file was created using the neural network trained on the entire dataset.