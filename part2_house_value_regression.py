import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute

# THIS IS TEMPORARY
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(13, 15)
        self.hidden = torch.nn.Linear(15, 5)
        self.output = torch.nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.input(x))
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.output(x)

        return x

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
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
        self.x_scaler = preprocessing.MinMaxScaler() # Perfoms min-max scaling on x values
        self.y_scaler = preprocessing.MinMaxScaler() # Performs min-max scaling on y values
        self.x_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean') # Used to handle empty cells
        self.lb = preprocessing.LabelBinarizer() # Used to handle ocean_proximity
        self.string_imp = None # Used to handle empty ocean_proximities

        self.net = Net()

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
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

        # CHANGE THIS (This works for now lol)
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        loss_metric = torch.nn.MSELoss()

        optimizer = torch.optim.SGD(self.net.parameters(), lr = 0.01)

        for _ in range(self.nb_epoch):
            self.net.zero_grad()
            output = self.net(X.float())
            loss = loss_metric(output, Y.float())
            loss.backward()
            optimizer.step()

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
        output = self.net(X.float())
        output = output.detach().numpy()

        return output


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
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

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
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

