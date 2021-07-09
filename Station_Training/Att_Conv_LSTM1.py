import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf

import keras
from keras_self_attention import SeqSelfAttention
from keras import backend as K

print(tf.__version__)

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(seed=42)
np.random.RandomState(42)
random.seed(42)
K.clear_session()

def train(x_train,y_train,x_test,y_test):


    os.environ['PYTHONHASHSEED'] = '42'
    np.random.seed(42)
    tf.random.set_seed(seed=42)
    np.random.RandomState(42)
    random.seed(42)

    model_lime_convlstm_att = keras.models.Sequential([
        keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 3),
                                strides=1,
                                activation="relu",
                                input_shape=(2, 1, int(x_train.shape[1] / 2), 1), return_sequences=True),
        # keras.layers.BatchNormalization(),
        keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 3),
                                strides=1,
                                activation="relu",
                                return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 3),
                                strides=1,
                                activation="relu",
                                return_sequences=True),
        # keras.layers.BatchNormalization(),

        keras.layers.Reshape((8, -1)),
        SeqSelfAttention(attention_activation='sigmoid'),
        keras.layers.Reshape((2, 1, 4, -1)),
        keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 3),
                                strides=1,
                                activation="relu",
                                return_sequences=True),
        keras.layers.BatchNormalization(),

        keras.layers.Reshape((4, -1)),
        SeqSelfAttention(attention_activation='sigmoid'),
        # keras.layers.Dropout(0.4),
        # keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        # keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        # keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation="relu"),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 400)
    ])

    optimizer = keras.optimizers.Adam(lr=1e-4)
    model_lime_convlstm_att.compile(loss='mse',
                                    optimizer=optimizer,
                                    metrics=["mae"])

    model_lime_convlstm_att.summary()
    # Create Model Checkpoints
    filepath_lime_convlstm_att = "weights_lime_convlstm_att_5P_soil.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath_lime_convlstm_att,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='min')

    history_lime_convlstm_att = model_lime_convlstm_att.fit(x_train, y_train,
                                                            validation_data=(x_test, y_test), epochs=10,
                                                            callbacks=[checkpoint])
    return model_lime_convlstm_att,filepath_lime_convlstm_att,history_lime_convlstm_att

def train_and_test(x_train,y_train,x_test,y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    keras.backend.clear_session()
    x_train_convlstm = x_train.reshape(x_train.shape[0], 2, 1, int(x_train.shape[1] / 2), 1)
    x_test_convlstm = x_test.reshape(x_test.shape[0], 2, 1, int(x_test.shape[1] / 2), 1)
    print(x_train_convlstm.shape)
    model_lime_convlstm_att,filepath_lime_convlstm_att,history_lime_convlstm_att=train(x_train_convlstm,y_train,x_test_convlstm,y_test)
    # Load best weight
    model_lime_convlstm_att.load_weights(filepath_lime_convlstm_att)

    plt.figure()
    plt.plot(np.arange(len(history_lime_convlstm_att.history['loss'])), history_lime_convlstm_att.history['loss'],
             color='r', label='Training loss')
    plt.plot(np.arange(len(history_lime_convlstm_att.history['loss'])), history_lime_convlstm_att.history['val_loss'],
             color='b', label='Validation loss')
    plt.title('Training vs Validation Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('ConvLSTM attention Lime 5P week(s) ahead - loss curve.jpg', quality=100, dpi=256, optimize=True)

    y_pred = model_lime_convlstm_att.predict(x_test_convlstm)
    y_pred_2x = y_pred

    plt.figure()
    plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual price')
    plt.plot(np.arange(len(y_test)), y_pred, color='b', label='Predicted price')
    plt.title('Actual vs Predicted Price')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Lime ACV-LSTM 5P week(s) ahead plot.jpg', quality=100, dpi=256, optimize=True)

    plt.figure()
    plt.plot(np.arange(len(y_test))[:50], y_test[:50], color='r', label='Actual price')
    plt.plot(np.arange(len(y_test))[:50], y_pred[:50], color='b', label='Predicted price')
    plt.title('Zoomed in Actual vs Predicted Yield')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Lime ACV-LSTM 5P week(s) ahead  zoomed plot.jpg', quality=100, dpi=256, optimize=True)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    agg_err = ((np.sqrt(mse) + mae) / 2) * (1 - r2)

    print('R2 Score: ', r2, ' , MAE: ', mae, ' , MSE: ', mse, ' , Agg: ', agg_err)