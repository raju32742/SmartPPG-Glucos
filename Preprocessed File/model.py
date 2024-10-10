import random
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from measrmnt_indices import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers


# def RFR():
#     rnr=RandomForestRegressor(**rfp)
#     return rnr


def RFR():
    # Fitting the Random Forest Regression Model to the dataset
#    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 150, random_state = 32)
    return regressor


import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np

# def reset_random_seeds():
#     os.environ['PYTHONHASHSEED']=str(1)
#     tf.random.set_seed(1)
#     np.random.seed(1)
#     random.seed(1)
# reset_random_seeds()


def DNN(X):
    model = Sequential()

    # The Input Layer :
    model.add(Dense(100, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    
    # The Hidden Layers :
    model.add(Dense(100, kernel_initializer='normal',activation='relu'))
    model.add(Dense(150, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(200, kernel_initializer='normal',activation='relu'))
    model.add(Dense(250, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    
    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network :
#    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MAE'])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    
    return model



# def DNN(X):
#     reset_random_seeds()

#     model = keras.Sequential([
#             keras.layers.Dense(100, kernel_initializer='normal', input_dim=X.shape[1], activation='relu'),
#             # The Hidden Layers :
#             keras.layers.Dense(150,  kernel_initializer='normal', activation='relu'),
#             keras.layers.Dense(200,  kernel_initializer='normal', activation='relu'),
#             keras.layers.Dropout(0.25),
#             keras.layers.Dense(250,  kernel_initializer='normal', activation='relu'),
#             keras.layers.Dropout(0.5),
#             # The Output Layer :
#             keras.layers.Dense(1, kernel_initializer='normal', activation='linear')
#         ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
#     return model  


def ANN(X):
    # reset_random_seeds()

    model = keras.Sequential([
            keras.layers.Dense(73, kernel_initializer='normal', input_dim=X.shape[1], activation='relu'),
            keras.layers.Dense(73,  kernel_initializer='normal', activation='relu'),
            keras.layers.Dense(1, kernel_initializer='normal', activation='linear')
        ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model   