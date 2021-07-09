import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import time
from math import sqrt
import warnings
import random

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Conv1D, Bidirectional, GRU, Flatten, Activation, \
    BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint


def rolling_window(val, r):
    x = []
    y = []
    for i in range(r, val.shape[0]):
        x_temp = val[i - r:i]
        y_temp = val[i]
        x.append(x_temp)
        y.append(y_temp)
    x = np.array(x)
    y = np.array(y)

    return x, y


def val_to_nan(features, labels, val_nan):
    features[features == val_nan] = np.nan
    labels[labels == val_nan] = np.nan

    return features, labels


def generate_missing(arr, percent, types):
    if types == 'Random':
        c = int(percent * arr.shape[0])
        print(f'{c} missing values will be generated')
        temp = arr.copy()
        np.random.seed(42)
        temp.ravel()[np.random.choice(temp.size, c, replace=False)] = np.nan
        return temp
    if types == 'Continous':
        random.seed(42)
        starting = random.randint(5, 20)
        c = int(percent * arr.shape[0])
        print(f'{c} missing values will be generated')
        temp = arr.copy()
        temp[starting:starting + c] = np.nan
        return temp


def individual_df(inputs, targets):
    assert type(inputs) == np.ndarray
    assert type(targets) == np.ndarray

    df_temp = (pd.concat([pd.DataFrame(inputs), pd.DataFrame(targets)], axis=1)).reset_index()
    df_temp.columns = ['Col' + str(j) for j in range(len(df_temp.columns))]

    df_temp_known = df_temp[df_temp.iloc[:, -1].notna()]
    print(f'The dataframe for known y is of shape {df_temp_known.shape}')
    #     before_impute = df_temp_known.iloc[:,1:-1].isnull().sum().sum()
    df_temp_known.interpolate(limit_direction='both', inplace=True)
    print(f'The dataframe for known y is interpolated')
    known_index = df_temp_known.iloc[:, 0].values
    known_inputs = df_temp_known.iloc[:, 1:-1].values
    known_targets = df_temp_known.iloc[:, -1].values

    assert known_index.shape[0] == known_inputs.shape[0]
    assert known_inputs.shape[0] == known_targets.shape[0]

    df_temp_unknown = df_temp[df_temp.iloc[:, -1].isna()]
    print(f'The dataframe for unknown y is of shape {df_temp_unknown.shape}')
    unknown_index = df_temp_unknown.iloc[:, 0].values
    unknown_inputs = df_temp_unknown.iloc[:, 1:-1].values
    unknown_targets = df_temp_unknown.iloc[:, -1].values

    assert unknown_index.shape[0] == unknown_inputs.shape[0]
    assert unknown_inputs.shape[0] == unknown_targets.shape[0]

    return (known_index, known_inputs, known_targets), (unknown_index, unknown_inputs, unknown_targets), df_temp


def general_model(params, input_data):
    layers_lstm = params['layers_lstm']
    neurons_lstm = params['neurons_lstm']
    layers_dense = params['layers_dense']
    neurons_dense = params['neurons_dense']

    model = Sequential()

    if layers_lstm == 1:
        model.add(LSTM(units=neurons_lstm[0], activation='relu',
                       return_sequences=False, input_shape=(input_data.shape[1], 1)))

    if layers_lstm > 1:
        model.add(LSTM(units=neurons_lstm[0], activation='relu',
                       return_sequences=True, input_shape=(input_data.shape[1], 1)))
        for i in range(1, layers_lstm):
            model.add(LSTM(units=neurons_lstm[i], activation='relu', return_sequences=True))
            model.add(Dropout(0.4))
    #         model.add(Dropout(0.4))
    model.add(Flatten())
    for j in range(layers_dense):
        #         print(j)
        model.add(Dense(units=neurons_dense[j], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    return model


def model_modifier(model, params, reset):
    layers_to_del = params['layers_to_del']
    layers_to_add = params['layers_to_add']
    nodes_to_add = params['nodes_to_add']
    layers_to_unfreeze = params['layers_to_unfreeze']

    if reset == True:
        for j in range(len(model.layers)):
            model.layers[j].trainable = True

    for l in range(len(model.layers)):
        model.layers[l].trainable = False

    layers = model.layers[:-layers_to_del]
    for i in range(layers_to_add):
        layers.append(Dense(nodes_to_add[i], activation='relu'))
    #         layers.append(Dropout(0.15))
    model_new = Sequential(layers)

    for unfreeze in range(layers_to_unfreeze):
        model_new.layers[-(unfreeze + 1)].trainable = True

    return model_new


def layers_freezer(model, params, reset):
    if reset == True:
        for j in range(len(model.layers)):
            model.layers[j].trainable = True
        print('The layers are all trainable now!!')
    layers_to_freeze = params['layers_to_freeze']
    for i in range(layers_to_freeze):
        model.layers[i].trainable = False

    return model


def metrics(actual, predicted):
    e_mae = mae(actual, predicted)
    e_mse = mse(actual, predicted)
    e_rmse = sqrt(e_mse)
    e_r2 = r2(actual, predicted)
    e_agg = (e_rmse + e_mae) / 2
    e_agm = (e_agg) * (1 - e_r2)

    return e_mae, e_mse, e_rmse, e_r2, e_agg, e_agm

