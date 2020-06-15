from keras.models import Sequential
from keras.layers import Dense, Input

from matplotlib import pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions

import keras.backend as K


def get_model():
    return Sequential([
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


def get_data():
    data = []
    y = []
    for i in range(-100, 100):
        data.append((i / 100.0, (i / 100.0) ** 2))
        y.append(0)
        
        if i % 10 == 0:
            data.append((i / 100.0, (i / 100.0) ** 2 - 1))
            y.append(1)
    
    return np.array(data), np.array(y)


def weighted_categorical_crossentropy(weights):
    """  
    A weighted version of keras.objectives.categorical_crossentropy
                                   
    Variables:
    weights: numpy array of shape (C,) where C is the number of classes
    """ 
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        return K.mean(
            K.binary_crossentropy(y_true, y_pred) * weights)
                                                                                                            
    return loss


model = get_model()
X, y = get_data()
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)
plot_decision_regions(X, y, clf=model, legend=2)


# Weighted loss
frac = sum(y) * 1.0 / len(y)
weights = np.array([1., 1. / frac])

model.compile('sgd', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
model.fit(X, y, epochs=10)
plot_decision_regions(X, y, clf=model, legend=2)