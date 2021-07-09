import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.decomposition import PCA
import joblib

def read_data_and_preprocessing(x,y,n,scl,pca):

    #fit1 = ExponentialSmoothing(y, seasonal_periods=12).fit()
    #y = fit1.fittedvalues
    y=np.asarray(y)
    x = x.values.reshape(len(x), 1)
    print(x.shape)
    x= series_to_supervised(x, 139)
    print(x.shape)
    print(y.shape)
    scaler = preprocessing.StandardScaler().fit(x)
    joblib.dump(scaler, scl)
    x= scaler.transform(x)
    k = Proportion_of_Variance(x)
    print('\n The value of d using POV = 95% is ', k)
    my_model = PCA(n_components=n)
    x = my_model.fit_transform(x)
    joblib.dump(my_model, pca)
    X = x
    Y = y
    Y = Y[:len(X)]
    print(X.shape)
    print(Y.shape)
    x_train = X[0:int(0.8 * X.shape[0])]
    x_test = X[int(0.8 * X.shape[0]):]
    y_train = Y[0:int(0.8 * Y.shape[0])]
    y_test = Y[int(0.8 * Y.shape[0]):]
    print(x_test.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(y_train.shape)
    return x_train,y_train,x_test,y_test


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#function to find the value of d for POV=98%
def Proportion_of_Variance(x):
    val,vec=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(np.abs(val))[::-1]
    val_sorted=val[indexes]
    val_sum=val_sorted.sum()
    for k in range(784):
        k_val_sum=val_sorted[:k+1].sum()
        POV=k_val_sum/val_sum
        if POV >= 0.991:
            break
    return(k+1)


