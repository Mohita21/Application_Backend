from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StationCodes import WebScraping as ws

df=pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Data/Soil_Mois_Temp_Imputed.xlsx")
df.index=df["LST_DATE"]
d=ws.scraping_current()
d1=ws.impute_current(df,d)
d1=ws.return_soil_mois_temp(d1)
#d1.to_excel("LatestMoistTemp.xlsx")
#print(d1)

DATE_UPPER=datetime.strptime("2011-01-01", '%Y-%m-%d')+timedelta(days=139)

def single_date(date):
    #date = datetime.strptime(date, '%Y-%m-%d')
    date_a=date
    date_b= date_a-timedelta(days=139)
    
    date_r=[date_b + timedelta(days=x) for x in range(140)]
    #print(date_r)
    #print("\n\n")
    #sm=[d1["Soil Moisture"][i] for i in date_r]
    try:
        st=[d1["Soil Temperature"][i] for i in date_r]

        st=np.asarray(st)
        st=st.reshape(1,st.shape[0])
        return list(st)
    except KeyError:
        print("Key error")

        #ft=[sm,st]
        #ft=np.asarray(ft)
        #ft=ft.reshape(1,ft.shape[1]*2)




def date_range(date1, date2,wa):
    if wa==0.5:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=2)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=2)
    if wa==1:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=7)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=7)
    if wa==2:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=14)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=14)
    if wa==3:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=21)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=21)
    if wa==4:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=28)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=28)
    if wa==5:
        date1 = datetime.strptime(date1, '%Y-%m-%d')-timedelta(days=35)
        date2 = datetime.strptime(date2, '%Y-%m-%d')-timedelta(days=35)
    if date1 < DATE_UPPER:
        #print("Select a date after",DATE_UPPER)
        raise ValueError("Select a date after",DATE_UPPER)
    delta=abs((date2 - date1).days)
    date_r=[date1 + timedelta(days=x) for x in range(delta+1)]
    #print("DATE .........",date_r)
    mt=[]
    for date in date_r:
        ft=single_date(date)
        mt.append(ft)
    mt=np.asarray(mt)
    print(mt.shape)
    mt=mt.reshape(mt.shape[0],mt.shape[2])
    return mt

def true_vals(sd,ed):
    date1 = datetime.strptime(sd, '%Y-%m-%d')
    date2 = datetime.strptime(ed, '%Y-%m-%d')
    delta=abs((date2 - date1).days)
    date_r=[date1 + timedelta(days=x) for x in range(delta+1)]
    d1=pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Ifeanyi_soil_weather_yield_price_5W.xlsx")
    d1.index=d1['Date']
    st = [d1["Yield"][i] for i in date_r]
    plt.plot(st,label='hi')
    plt.legend()
    plt.show()
    return st

#date_range('2021-04-01','2021-04-05',5)
#true_vals('2019-04-01','2019-05-01')
#single_date('2021-04-01')

#single_date('2021-04-02')
