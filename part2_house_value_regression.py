import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute, metrics, model_selection
from sklearn.base import BaseEstimator

class Regressor():

    def __init__(self, x, nb_epoch = 1000, neurons = [8, 8, 1], learning_rate = 0.1, loss_fun = "mse"):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Values stored for pre-processing
        self.x = x
        self.x_scaler = preprocessing.MinMaxScaler() # Perfoms min-max scaling on x values
        self.y_scaler = preprocessing.MinMaxScaler() # Performs min-max scaling on y values
        self.x_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean') # Used to handle empty cells
        self.lb = preprocessing.LabelBinarizer() # Used to handle ocean_proximity
        self.string_imp = None # Used to handle empty ocean_proximities

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Initialising Net stuff
        self.neurons = neurons # Architecture of net
        layers = []
        n_in = self.input_size
        for layer in neurons:
            layers.append(nn.Linear(n_in, layer)) # Use Linear activation functions only
            n_in = layer


        layers.append(nn.ReLU()) # Use ReLU as final activation function
        
        self.net = nn.Sequential(*layers) # Stack-Overflow Bless
        self.learning_rate = learning_rate

        if loss_fun == "mse":
            self.loss_layer = nn.MSELoss()
        else:
            raise Exception(f'Undefined loss_fun: {loss_fun}')
        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # First we handle the strings
        # Deal with empty cells in string
        if training: self.string_imp = x['ocean_proximity'].mode()[0]
        pd.options.mode.chained_assignment = None
        x['ocean_proximity'] = x.loc[:, ['ocean_proximity']].fillna(value=self.string_imp)

        # Replace strings with binary values
        if training: self.lb.fit(x['ocean_proximity'])
        proximity = self.lb.transform(x['ocean_proximity'])
        x = x.drop('ocean_proximity', axis=1)
        x = pd.concat([x, pd.DataFrame(proximity)], axis=1)

        # Next we impute (deal with empty cells)
        if training: self.x_imp.fit(x)

        x = self.x_imp.transform(x)

        # If training we initialise our normalisation values
        if training:
            self.x_scaler.fit(x)
            if isinstance(y, pd.DataFrame): self.y_scaler.fit(y)

        x = torch.from_numpy(self.x_scaler.transform(x))
        y = torch.from_numpy(self.y_scaler.transform(y)) if isinstance(y, pd.DataFrame) else None
        
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

        for i in range(self.nb_epoch):
            optimizer.zero_grad()
            output = self.net(X.float())
            loss = self.loss_layer(output, Y.float())
            loss.backward()
            optimizer.step()
            print(f'Loss at epoch {i}: {self.score(x, y)}')

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        output = self.net(X.float()).detach().numpy()

        return self.y_scaler.inverse_transform(output)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        output = self.net(X.float()).detach().numpy()

        y_hat = self.y_scaler.inverse_transform(output)
        y_gold = y.to_numpy()
        return metrics.mean_squared_error(y_gold, y_hat, squared=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def get_params(self, deep=True):
        return {
            'x': self.x,
            'learning_rate': self.learning_rate,
            'nb_epoch': self.nb_epoch,
            'neurons': self.neurons
        }

    def set_params(self, **params):
        for param, value in params.items():
            if param == 'neurons':
                layers = []
                n_in = self.input_size
                for layer in neurons:
                    layers.append(nn.Linear(n_in, layer)) # Use Linear activation functions only
                    n_in = layer

                layers.append(nn.ReLU()) # Use ReLU as final activation function
                self.net = nn.Sequential(*layers) # Stack-Overflow Bless

            setattr(self, param, value)


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(regressor, x, y, params): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
        - params {dictionary} -- Dictionary with parameter names (str) as keys and lists of parameter settings to try as values
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    X, Y = regressor._preprocessor(x, y = y, training = False) # Do not forget
    scorer = metrics.make_scorer(regressor.score, greater_is_better=False)


    gs = model_selection.GridSearchCV(
        regressor, 
        params, 
        n_jobs=1, # Set n_jobs to -1 for parallelisation
        refit=True, 
        verbose=2, 
        return_train_score=True)

    gs.fit(X, Y)

    return  gs.best_params

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

