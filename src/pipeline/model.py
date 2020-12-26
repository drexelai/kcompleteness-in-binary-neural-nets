# Author: Isamu Isozaki, Yigit Alparslan
# Date: 2020/11/10
# Purpose: construct model given input hidden_layers

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

# Setup model given hidden layer with dimension input dimension*i for i in hidden_layers
def create_model_from_given_architecture(**args):
    model = Sequential()
    model.add(BatchNormalization())
    hidden_layers = args['hidden_layers']
    for layer_number in range(len(hidden_layers)):
        print(layer_number, hidden_layers, "#############################################")
        if layer_number == 0: 
            # Add input layer
            model.add(Dense(hidden_layers[layer_number], input_dim=11, activation='relu'))
        else:
            # Add hidden layers
            model.add(Dense(hidden_layers[layer_number], activation = 'relu'))
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    return model