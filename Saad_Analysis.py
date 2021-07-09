import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import numpy as np


def print_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors


def trend_determiner(values):
    split = int(values.shape[0] / 2)
    x1 = values[0:split]
    x2 = values[split:]

    if x1.mean() <= 2 and x2.mean() <= 2:
        x1 = x1 * 100
        x2 = x2 * 100

    difference = abs(np.divide(x1.mean() - x2.mean(), x1.mean()))
    print(x1.mean(), x2.mean())

    if difference > 0.1:
        return 1
    else:
        return 0


def prime_numbers(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def seasonal_determiner(values, f):
    yhat = abs(savgol_filter(values, f, 7))
    rangee = values.max() - values.min()
    if (np.mean(abs(values) - yhat)) / rangee < 1:
        return 1
    else:
        return 0


def Analyzer(data):
    c = 0
    data=np.asarray(data)
    flag=1
    data=data.reshape(data.shape[0])
    for i in range(len(data)):
        #print(data.shape)
        #print("iloc",data[i])
        if not (data[i]==-1):
            c = 0
        elif (data[i] ==-1):
            c += 1
            #print(c, i)
        if c > 20:
            #print('Data contains missing chunks (more than 10 consecutive missing values) thus cannot be imputed.')
            flag=0
            #print("More than 10 chunks")
    data=pd.DataFrame(data)
    #print(data)

    data = data.replace(0, np.nan)
    data = data.replace(-1, np.nan)
    data = data.replace("nan", np.nan)
    print("This",c)
    print(data)
    try:
        f = 365
        de_brent = seasonal_decompose(data.dropna(), model='multiplicative', period=f)
    except Exception as e:
        f = 13
        de_brent = seasonal_decompose(data.dropna(), model='multiplicative', period=f)
        #print(e)

    trend_result = trend_determiner(de_brent.trend[~np.isnan(de_brent.trend)])
    seasonal_result = seasonal_determiner(de_brent.seasonal, f)

    if trend_result == 1 and seasonal_result == 1:
        result = 'both Trend and Seasonality'
    elif trend_result == 0 and seasonal_result == 1:
        result = 'Seasonality'
    elif trend_result == 1 and seasonal_result == 0:
        result = 'Trend'
    else:
        result = 'Random'

    return result,flag


#df = pd.read_csv('Orange-San Diego- Coachella-finalized.csv',  thousands=',')
'''
df = pd.read_csv('/Users/mohita/Desktop/Files for App/miss3.csv')
print("This",Analyzer(df))
'''