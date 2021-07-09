from urllib.request import urlopen
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def yearly_data_save(html):
    st = ""
    i = 0
    c = html[i]
    w = []
    l = []
    while (i < len(html)):
        while (c != " " and c != "\n" and i < len(html)):
            st = st + c
            c = html[i]
            i = i + 1
        w.append(st)
        st = ""
        if c == "\n":
            l.append(w)
            w = []
        while (c == " " and i < len(html)):
            c = html[i]
            i = i + 1

        while (c == "\n" and i < len(html)):
            c = html[i]
            i = i + 1
    return l


def final_frame():
    today = datetime.today()
    frames = []
    for i in range(2011, today.year + 1):
        url = "https://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01/" + str(i) + "/CRND0103-" + str(
            i) + "" + "-CA_Santa_Barbara_11_W.txt"
        page = urlopen(url)
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")
        d = yearly_data_save(html)
        print("The data in the year", i, "is", len(d), "long")
        d = pd.DataFrame(d)
        frames.append(d)
    df = pd.concat(frames)
    Generating_dates_and_columns(df)
    return df


def Generating_dates_and_columns(s):
    s.columns = ["WBANNO", "LST_DATE", "CRX_VN", "LONGITUDE", "LATITUDE", "T_DAILY_MAX", "T_DAILY_MIN", "T_DAILY_MEAN",
                 "T_DAILY_AVG",
                 "P_DAILY_CALC", "SOLARAD_DAILY", "SUR_TEMP_DAILY_TYPE", "SUR_TEMP_DAILY_MAX", "SUR_TEMP_DAILY_MIN",
                 "SUR_TEMP_DAILY_AVG",
                 "RH_DAILY_MAX", "RH_DAILY_MIN", "RH_DAILY_AVG", "SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
                 "SOIL_MOISTURE_20_DAILY",
                 "SOIL_MOISTURE_50_DAILY", "SOIL_MOISTURE_100_DAILY", "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY",
                 "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY",
                 "SOIL_TEMP_100_DAILY"]
    s.index = np.arange(0, len(s))
    for i in range(len(s)):
        da = s['LST_DATE'][i]
        da = str(da)
        d = datetime(year=int(da[0:4]), month=int(da[4:6]), day=int(da[6:8]))
        s['LST_DATE'][i] = d
    s.index = s["LST_DATE"]





def Mois_and_Temp_Imputation(s):
    col=["SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
                 "SOIL_MOISTURE_20_DAILY",
                 "SOIL_MOISTURE_50_DAILY", "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY",
                 "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY"]
    for col_name in col:
        for i in range(len(s)):
            if float(s[col_name][i]) < 0:
                s[col_name][i] = np.nan
        #plt.plot(pd.to_numeric(s[col_name]))
        #plt.title("Data before Imputation:" + col_name)
        #plt.show()
        for i in range(len(s)):
            if np.isnan(float(s[col_name][i])):
                # print("yes")
                y = s['LST_DATE'][i] + relativedelta(years=1)
                z = s['LST_DATE'][i] - relativedelta(years=1)
                for k in range(len(s)):
                    if s['LST_DATE'][k] == y and not (np.isnan(float(s[col_name][k]))):
                        s[col_name][i] = s[col_name][k]
                    elif s['LST_DATE'][k] == z and not (np.isnan(float(s[col_name][k]))):
                        s[col_name][i] = s[col_name][k]

        #plt.plot(pd.to_numeric(s[col_name]))
        #plt.title("Data after Imputation:" + col_name)
        #plt.show()
        #print("Any missing values", np.isnan(pd.to_numeric(s[col_name]).values).any())


def return_soil_mois_temp(df):
    Mois_and_Temp_Imputation(df)
    df["Soil Moisture"]=""
    df["Soil Temperature"]=""
    for i in range(len(df)):
        df["Soil Moisture"][i]=float(0.3) *float(df["SOIL_MOISTURE_5_DAILY"][i]) + float(0.3) *float(df["SOIL_MOISTURE_10_DAILY"][i]) + float(0.3) *float(df["SOIL_MOISTURE_20_DAILY"][i])  +float(0.1) *float(df["SOIL_MOISTURE_50_DAILY"][i])
        df["Soil Temperature"][i]=float(0.3) *float(df["SOIL_TEMP_5_DAILY"][i]) + float(0.3) *float(df["SOIL_TEMP_10_DAILY"][i]) +float(0.3) *float(df["SOIL_TEMP_20_DAILY"][i])  +float(0.1) *float(df["SOIL_TEMP_50_DAILY"][i])

    return df.drop(["WBANNO", "LST_DATE", "CRX_VN", "LONGITUDE", "LATITUDE", "T_DAILY_MAX", "T_DAILY_MIN", "T_DAILY_MEAN",
                 "T_DAILY_AVG",
                 "P_DAILY_CALC", "SOLARAD_DAILY", "SUR_TEMP_DAILY_TYPE", "SUR_TEMP_DAILY_MAX", "SUR_TEMP_DAILY_MIN",
                 "SUR_TEMP_DAILY_AVG",
                 "RH_DAILY_MAX", "RH_DAILY_MIN", "RH_DAILY_AVG","SOIL_TEMP_100_DAILY","SOIL_MOISTURE_100_DAILY","SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
                 "SOIL_MOISTURE_20_DAILY",
                 "SOIL_MOISTURE_50_DAILY", "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY",
                 "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY"],axis=1)

def scraping_current():
    
    today = datetime.today()
    i=today.year
    #frames = []
    #for i in range(2011, today.year + 1):
    url = "https://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01/" + str(i) + "/CRND0103-" + str(
            i) + "" + "-CA_Santa_Barbara_11_W.txt"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    d = yearly_data_save(html)
    print("The data in the year", i, "is", len(d), "long")
    d = pd.DataFrame(d)
    #frames.append(d)
    #df = pd.concat(frames)
    Generating_dates_and_columns(d)
    return d

def impute_current(df,d):
    d1 = pd.concat([df,d],sort=False)
    #print(d1)
    col=["SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
                 "SOIL_MOISTURE_20_DAILY",
                 "SOIL_MOISTURE_50_DAILY", "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY",
                 "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY"]
    for col_name in col:
        for i in range(len(df),(len(df)+len(d)-1)):
            if float(d1[col_name][i]) < 0:
                d1[col_name][i] = np.nan
        #plt.plot(pd.to_numeric(d1[col_name]))
        #plt.title("Data before Imputation:" + col_name)
        #plt.show()
        for i in range(len(df),(len(df)+len(d)-1)):
            if np.isnan(float(d1[col_name][i])):
                # print("yes")
                y = d1['LST_DATE'][i] + relativedelta(years=1)
                z = d1['LST_DATE'][i] - relativedelta(years=1)
                for k in range(len(d1)):
                    if d1['LST_DATE'][k] == y and not (np.isnan(float(d1[col_name][k]))):
                        d1[col_name][i] = d1[col_name][k]
                    elif d1['LST_DATE'][k] == z and not (np.isnan(float(d1[col_name][k]))):
                        d1[col_name][i] = d1[col_name][k]

        #plt.plot(pd.to_numeric(d1[col_name]))
        #plt.title("Data after Imputation:" + col_name)
        #plt.show()
        #print("Any missing values", np.isnan(pd.to_numeric(d1[col_name]).values).any())
    return d1

#df=final_frame()
#df=Mois_and_Temp_Imputation(df)