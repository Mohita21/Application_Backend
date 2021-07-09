import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Imputation import TF_Utils as tfu
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

'''
Assumptions:

* The complete time series has only one column which has values of time series
* The incomplete time series has only one column which has missing values in it
* The file will be saved to the current directory, in order to change it kindly alter the line 125 (df_combined.to_excel)
'''


def merger(complete, incomplete):
    '''
    The function merges two excel files and returns one dataframe

    Params:
    complete : The complete is the filepath to the complete timeseries
    incomplete: The incomplete is the filepath to the incomplete timeseries

    Return:
    df_combined : The dataframe which is the combined version of two excel files.
    '''

    #df_complete = pd.read_csv(complete,header=None,skip_blank_lines=False)
    #df_incomplete = pd.read_csv(incomplete,header=None,skip_blank_lines=False)
    df_complete= pd.DataFrame(complete)
    df_incomplete= pd.DataFrame(incomplete)
    print(len(df_incomplete))
    print(len(df_complete))
    #print("This",df_incomplete.isnull().any())
    #assert df_complete.isnull().sum().values[0] == 0, 'The complete time series is not complete'
    #assert df_incomplete.isnull().sum().values[0] != 0, 'The incomplete time series has no missing rows'

    df_combined = pd.concat([df_complete, df_incomplete], axis=1)
    df_combined.columns = ['Reference', 'Target']
    return df_combined


def Imputer(df):
    '''
    The function imputes the chunks in the dataframe

    Params:
    df: The dataframe which is has couple of columns. The first column has complete timeseries whereas the
        second column has missing values.

    Return:

    '''
    # Transfer Learning Imputation
    df_missing = df[['Target']]
    df_complete = df[['Reference']]
    scaler_incomplete = MinMaxScaler(feature_range=(0, 1))
    scaler_complete = MinMaxScaler(feature_range=(0, 1))
    df_incomplete_scaled = pd.DataFrame(scaler_incomplete.fit_transform(df_missing), columns=df_missing.columns)
    df_complete_scaled = pd.DataFrame(scaler_complete.fit_transform(df_complete), columns=df_complete.columns)

    ROLLING_WINDOW = 3

    temp_missing = np.nan_to_num(df_incomplete_scaled.values, nan=-999)
    miss_inputs, miss_targets = tfu.rolling_window(temp_missing, ROLLING_WINDOW)
    miss_inputs = miss_inputs[:, :, 0]

    miss_inputs, miss_targets = tfu.val_to_nan(miss_inputs, miss_targets, -999)

    known_y, unknown_y, complete_df = tfu.individual_df(miss_inputs, miss_targets)

    known_y_index, known_y_inputs, known_y_target = known_y[0], known_y[1], known_y[2]

    params = dict(layers_lstm=4, layers_dense=2, neurons_lstm=[64, 64, 64, 64], neurons_dense=[64, 18])

    base_inputs, base_targets = tfu.rolling_window(df_complete_scaled.values, ROLLING_WINDOW)
    base_inputs = base_inputs.reshape(base_inputs.shape[0], base_inputs.shape[1], 1)

    filepath = "pretrained_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    pre_trained = tfu.general_model(params, base_inputs)

    pre_trained.compile(loss='mse', optimizer='adam', metrics=['mae'])
    history_pretrained = pre_trained.fit(base_inputs, base_targets, epochs=200, validation_split=0.2,
                                         callbacks=[checkpoint])

    model_interim = tfu.general_model(params, base_inputs)
    model_interim.load_weights(filepath)
    params['layers_to_del'] = 1
    params['layers_to_add'] = 1
    params['nodes_to_add'] = [1]
    params['layers_to_unfreeze'] = 11

    model_target = tfu.model_modifier(model_interim, params, False)
    model_target.summary()

    model_target.compile(loss='mse', optimizer='adam', metrics=['mae'])
    known_y_inputs = known_y_inputs.reshape(known_y_inputs.shape[0], known_y_inputs.shape[1], 1)

    filepath_interim = "interim.h5"
    checkpoint_interim = ModelCheckpoint(filepath_interim, monitor='val_loss',
                                         verbose=0, save_best_only=True, mode='min')

    history_target = model_target.fit(known_y_inputs, known_y_target, epochs=100,
                                      validation_split=0.2, callbacks=[checkpoint_interim])
    x_complete = np.copy(complete_df.iloc[:, 1:-1].values)
    y_complete = np.copy(complete_df.iloc[:, -1].values)

    for c in range(10):
        for i in range(x_complete.shape[0]):
            if np.isnan(y_complete[i]):
                x_temp = x_complete[i].reshape(-1, 1)
                x_temp[0] = base_inputs[i][0]
                x_temp[ROLLING_WINDOW - 1] = base_inputs[i][ROLLING_WINDOW - 1]
                x_temp = x_temp.reshape(x_temp.shape[1], x_temp.shape[0], 1)
                temp = model_target.predict(x_temp.reshape(x_temp.shape[0], x_temp.shape[1], 1))
                np.fill_diagonal(np.fliplr(x_complete[i + 1:]), temp)
                y_complete[i] = temp
    complete = np.concatenate([x_complete, y_complete.reshape(-1, 1)], axis=1)
    imputed_df = pd.DataFrame(complete, columns=complete_df.columns[1:])
    imputed_dataset = np.concatenate([imputed_df.iloc[0].values, imputed_df.iloc[1:, -1].values], axis=0)
    foo = scaler_incomplete.inverse_transform(imputed_dataset.reshape(-1, 1))
    foo1 = scaler_incomplete.inverse_transform(df_incomplete_scaled)
    imputed_dataframe = pd.DataFrame(foo, columns=['Imputed'])
    df_combined = pd.concat([df, imputed_dataframe], axis=1)
    plt.plot(df_combined)
    plt.show()
    df_combined.to_excel('/Users/mohita/Documents/GitHub/Flask_app/Data/Transfer Learning Results.xls')
    os.remove('interim.h5')
    os.remove('pretrained_model.h5')

    return

'''
if __name__ == '__main__':
    complete_file = '/Users/mohita/Desktop/Files for App/complete_TS.csv'
    incomplete_file = '/Users/mohita/Desktop/Files for App/Incomplete_TS.csv'
    combined_file = merger(complete_file, incomplete_file)
    Imputer(combined_file)

'''