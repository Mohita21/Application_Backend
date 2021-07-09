import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from Station_Training import Data_preprocessing as dp
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

df=pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Data/LatestMoistTemp.xlsx")
X = df["Soil Temperature"]

def TL_out(model_lime_cnn_lstm_att,weight_path,test,Y,n):
    model_lime_cnn_lstm_att.load_weights(weight_path)
    scl = "sclrWeights/TL_sclr_yield_5W.pkl"
    pca = "pcaWeights/TL_pca_yield_5W.pkl"
    df = pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Data/LatestMoistTemp.xlsx")
    X = df["Soil Temperature"]
    print("Type", type(X))
    m_l= min(len(X),len(Y))
    X = X[:m_l]
    Y= Y[:m_l]

    x_train, y_train, x_test, y_test = dp.read_data_and_preprocessing(X, Y, n, scl, pca)
    model_lime_cnn_lstm_att.trainable = True

    path_ft = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/Weights_FT.hdf5"

    checkpoint = ModelCheckpoint(path_ft,monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='min')
    inputs = keras.Input(shape=(n, 1))

    x = model_lime_cnn_lstm_att(inputs)
    x1 = keras.layers.Dense(64, activation="relu")(x)
    x2= keras.layers.Dense(32, activation="relu")(x1)
    x3 = keras.layers.Dense(16, activation="relu")(x2)

    outputs = keras.layers.Dense(1)(x3)

    # outputs=keras.layers.Lambda(lambda x: x * 400)(x)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    plt.figure()
    plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], color='r', label='Training loss')
    plt.plot(np.arange(len(history.history['loss'])), history.history['val_loss'], color='b', label='Validation loss')
    plt.title('Training vs Validation Loss (MAE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    y_pred = model.predict(x_test)

    plt.figure()
    plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual yield')
    plt.plot(np.arange(len(y_test)), y_pred, color='k', label='Predicted yield')
    plt.title('Actual vs Predicted Price')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    agg_err = ((np.sqrt(mse) + mae) / 2) * (1 - r2)

    print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ', np.sqrt(mse), ' , Agg: ', agg_err)
    scaler = joblib.load(scl)
    test = scaler.transform(test)
    my_model = joblib.load(pca)
    test = my_model.transform(test)
    test = test.reshape(test.shape[0], test.shape[1], 1)
    prediction= model.predict(test)
    prediction = prediction.tolist()
    prediction = ['%.2f' % elem[0] for elem in prediction]

    return prediction

def TL_in(model_lime_cnn_lstm_att,weight_path,test,Y,X,n):
    print(X)
    print("Type",type(X))
    model_lime_cnn_lstm_att.load_weights(weight_path,)
    scl = "sclrWeights/TL_sclr_yield_5W.pkl"
    pca = "pcaWeights/TL_pca_yield_5W.pkl"

    x_train, y_train, x_test, y_test = dp.read_data_and_preprocessing(X, Y, n, scl, pca)
    model_lime_cnn_lstm_att.trainable = True

    path_ft = "../TrainingWeights/Weights_FT.hdf5"

    checkpoint = ModelCheckpoint(path_ft,monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='min')
    inputs = keras.Input(shape=(n, 1))

    x = model_lime_cnn_lstm_att(inputs)
    x1 = keras.layers.Dense(64, activation="relu")(x)
    x2= keras.layers.Dense(32, activation="relu")(x1)
    x3 = keras.layers.Dense(16, activation="relu")(x2)

    outputs = keras.layers.Dense(1)(x3)

    # outputs=keras.layers.Lambda(lambda x: x * 400)(x)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    plt.figure()
    plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], color='r', label='Training loss')
    plt.plot(np.arange(len(history.history['loss'])), history.history['val_loss'], color='b', label='Validation loss')
    plt.title('Training vs Validation Loss (MAE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    y_pred = model.predict(x_test)

    plt.figure()
    plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual yield')
    plt.plot(np.arange(len(y_test)), y_pred, color='k', label='Predicted yield')
    plt.title('Actual vs Predicted Price')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    agg_err = ((np.sqrt(mse) + mae) / 2) * (1 - r2)

    print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ', np.sqrt(mse), ' , Agg: ', agg_err)
    scaler = joblib.load(scl)
    test = scaler.transform(test)
    my_model = joblib.load(pca)
    test = my_model.transform(test)
    test = test.reshape(test.shape[0], test.shape[1], 1)
    prediction= model.predict(test)
    prediction = prediction.tolist()
    prediction = ['%.2f' % elem[0] for elem in prediction]

    return prediction