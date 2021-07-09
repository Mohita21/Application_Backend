
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import pickle
import tensorflow as tf
import tensorflow
import joblib

import keras
from keras_self_attention import SeqSelfAttention
from keras import backend as K

print(tf.__version__)
#print(keras.__version__)

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(seed=42)
np.random.RandomState(42)
random.seed(42)
K.clear_session()


def model(n):
    keras.backend.clear_session()
    os.environ['PYTHONHASHSEED'] = '42'
    np.random.seed(42)
    tf.random.set_seed(seed=42)
    np.random.RandomState(42)
    random.seed(42)
    model_lime_cnn_lstm_att = keras.models.Sequential()

    model_lime_cnn_lstm_att.add(keras.layers.Conv1D(filters=120, kernel_size=3,strides=1,
                            padding="causal",activation="relu",
                            input_shape=(n, 1)))
    model_lime_cnn_lstm_att.add(keras.layers.Conv1D(filters=120, kernel_size=3,
                            strides=1, padding="causal",
                            activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.Conv1D(filters=120, kernel_size=3,
                            strides=1, padding="causal",
                            activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.Conv1D(filters=120, kernel_size=3,
                            strides=1, padding="causal",
                            activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.LSTM(100, return_sequences=True, activation='relu'))
    model_lime_cnn_lstm_att.add(keras.layers.LSTM(100, return_sequences=True, activation='relu'))
    model_lime_cnn_lstm_att.add(SeqSelfAttention(attention_activation='sigmoid'))
    model_lime_cnn_lstm_att.add(keras.layers.Reshape((12,100)))
    model_lime_cnn_lstm_att.add(keras.layers.Flatten())
    model_lime_cnn_lstm_att.add(keras.layers.Dense(64, activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.Dense(32, activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.Dense(16, activation="relu"))
    model_lime_cnn_lstm_att.add(keras.layers.Dense(1))
    model_lime_cnn_lstm_att.add(keras.layers.Lambda(lambda x: x * 400))


    optimizer = keras.optimizers.Adam(lr=1e-4)
    model_lime_cnn_lstm_att.compile(loss='mae',
                                    optimizer=optimizer,
                                    metrics=["mse"])
    return model_lime_cnn_lstm_att


def train(model_lime_cnn_lstm_att,x_train,y_train,x_test,y_test,path):
    # Create Model Checkpoints
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    filepath_lime_cnn_lstm_att = path
    checkpoint = keras.callbacks.ModelCheckpoint(filepath_lime_cnn_lstm_att,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min')
    history_lime_cnn_lstm_att = model_lime_cnn_lstm_att.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1,
                                                            epochs=400,callbacks=[checkpoint])
    #pickle.dump(model_lime_cnn_lstm_att, open('model22.pkl','wb'))
    return model_lime_cnn_lstm_att,filepath_lime_cnn_lstm_att,history_lime_cnn_lstm_att




def test(model_lime_cnn_lstm_att,filepath_lime_cnn_lstm_att,history_lime_cnn_lstm_att,x_test,y_test):
    # Load best weights

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    model_lime_cnn_lstm_att.load_weights(filepath_lime_cnn_lstm_att)

    plt.figure()
    plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['loss'],
             color='r', label='Training loss')

    plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['val_loss'],
             color='b', label='Validation loss')
    plt.title('Training vs Validation Loss (MAE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    #plt.savefig('CNN-LSTM attention Lime 5P week(s) ahead - loss curve.jpg', quality=100, dpi=256, optimize=True)

    y_pred = model_lime_cnn_lstm_att.predict(x_test)
    #y_pred_1x = y_pred

    plt.figure()
    plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual price')
    plt.plot(np.arange(len(y_test)), y_pred, color='b', label='Predicted price')
    plt.title('Actual vs Predicted Price')
    plt.ylabel('Price')
    plt.legend()
    #plt.savefig('Lime AC-LSTM 5P week(s) ahead plot.jpg', quality=100, dpi=256, optimize=True)
    plt.show()



    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    agg_err = ((np.sqrt(mse) + mae) / 2) * (1 - r2)

    print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ', np.sqrt(mse), ' , Agg: ', agg_err)

def test_train_data(model_lime_cnn_lstm_att,filepath_lime_cnn_lstm_att,x_test,y_test):
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    model_lime_cnn_lstm_att.load_weights(filepath_lime_cnn_lstm_att)
    y_pred = model_lime_cnn_lstm_att.predict(x_test)
    #y_pred_1x = y_pred

    plt.figure()
    plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual price')
    plt.plot(np.arange(len(y_test)), y_pred, color='b', label='Predicted price')
    plt.title('Actual vs Predicted Price')
    plt.ylabel('Price')
    plt.legend()
    #plt.savefig('Lime AC-LSTM 5P week(s) ahead plot.jpg', quality=100, dpi=256, optimize=True)
    plt.show()



    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    agg_err = ((np.sqrt(mse) + mae) / 2) * (1 - r2)

    print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ', np.sqrt(mse), ' , Agg: ', agg_err)


def transfer_learning(n,filepath_lime_cnn_lstm_att,x_train,y_train,x_test,y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    model_lime_cnn_lstm_att=model(n)
    model_lime_cnn_lstm_att.load_weights(filepath_lime_cnn_lstm_att)
    final_model,updated_path, history=train(model_lime_cnn_lstm_att,x_train,y_train,x_test,y_test)
    test(final_model, updated_path, history,x_test,y_test)
    return final_model,updated_path, history








