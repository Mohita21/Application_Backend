# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:09:01 2020

@author: MohamedSadok
"""

from statsmodels.tsa.api import ExponentialSmoothing
import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import matplotlib.pyplot as plt
import pdb

import tifffile
# import gdal
# gdal.UseExceptions()

import ee

# ee.Authenticate()

import time

import numpy as np
import pandas as pd
import os
import urllib

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import ModelCheckpoint

tf.keras.backend.clear_session()


def appendTemp(current, previous):
    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0, 4])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum


def appendMoist(current, previous):
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(['ssm', 'susm'])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum


def appendMask(current, previous):
    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum


def ymdList(imgcol):
    def iter_func(image, newlist):
        date = ee.Number.parse(image.date().format("YYYYMMdd"));
        newlist = ee.List(newlist);
        return ee.List(newlist.add(date).sort())

    ymd = imgcol.iterate(iter_func, ee.List([]))
    return list(ee.List(ymd).reduce(ee.Reducer.frequencyHistogram()).getInfo().keys())


def getHists(train_date_start, train_date_end, fips, horizon=0):
    ''
    ### Starting Google Earth Engine ###
    ee.Initialize()

    state = fips[:2]
    county = fips[2:]
    ### Temperature ###
    imgcoll = ee.ImageCollection('MODIS/006/MOD11A1').filterBounds(ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate(
        str(train_date_start), str(train_date_end + datetime.timedelta(days=1)))
    img = ee.Image(imgcoll.iterate(appendTemp))
    # [Santa Maria in Santa Barbara]
    # offset = 0.11
    scale = 500
    crs = 'EPSG:3857'

    region = ee.FeatureCollection('TIGER/2018/Counties')
    region = region.filterMetadata('STATEFP', 'equals', state).filterMetadata('COUNTYFP', 'equals', county)
    region = ee.Feature(region.first())
    filename = 'temp'
    task = ee.batch.Export.image(img.clip(region), filename,
                                 {'driveFolder': 'Satellite Images', 'driveFileNamePrefix': filename, 'scale': scale,
                                  'crs': crs})
    task.start()
    print('Done.', task.status())

    ### Moisture ###
    imgcoll = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture').filterBounds(
        ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate(str(train_date_start),
                                                             str(train_date_end + datetime.timedelta(days=1)))
    img = ee.Image(imgcoll.iterate(appendMoist))
    # filter for a county
    region = ee.FeatureCollection('TIGER/2018/Counties')
    region = region.filterMetadata('STATEFP', 'equals', state).filterMetadata('COUNTYFP', 'equals', county)
    region = ee.Feature(region.first())
    filename = 'moist'
    task = ee.batch.Export.image(img.clip(region), filename,
                                 {'driveFolder': 'Satellite Images', 'driveFileNamePrefix': filename, 'scale': scale,
                                  'crs': crs})
    task.start()
    print('Done.', task.status())

    ### Land Cover Mask ###
    if int(train_date_start.year) >= 2019 and int(train_date_end.year) > 2019:
        imgcoll = ee.ImageCollection('MODIS/006/MCD12Q1').filterBounds(
            ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate('2019-01-01', '2019-01-02')
    elif int(train_date_start.year) < 2019 and int(train_date_end.year) >= 2019:
        imgcoll = ee.ImageCollection('MODIS/006/MCD12Q1').filterBounds(
            ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate(str(train_date_start.year) + '-01-01', '2019-01-02')
    elif int(train_date_end.year) <= 2019:
        imgcoll = ee.ImageCollection('MODIS/006/MCD12Q1').filterBounds(
            ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate(str(train_date_start.year) + '-01-01',
                                                                 str(train_date_end + datetime.timedelta(days=1)))
    img = ee.Image(imgcoll.iterate(appendMask))
    # filter for a county
    region = ee.FeatureCollection('TIGER/2018/Counties')
    region = region.filterMetadata('STATEFP', 'equals', state).filterMetadata('COUNTYFP', 'equals', county)
    region = ee.Feature(region.first())
    filename = 'mask'
    task = ee.batch.Export.image(img.clip(region), filename,
                                 {'driveFolder': 'Satellite Images', 'driveFileNamePrefix': filename, 'scale': scale,
                                  'crs': crs})
    task.start()
    print('Done.', task.status())

    ### Downloading images from Google Drive ###
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file in fileList:
        if file['title'] == 'Satellite Images':
            folderID = file['id']

    fileList = drive.ListFile({'q': "'" + folderID + "' in parents and trashed=false"}).GetList()
    for file in fileList:
        print(file['title'], 'old file deleted')
        file.Delete()
        time.sleep(1)

    while len(fileList) < 3:
        fileList = drive.ListFile({'q': "'" + folderID + "' in parents and trashed=false"}).GetList()
        time.sleep(10)

    for file in fileList:
        print(file['title'])
        file.GetContentFile(file['title'])
        file.Delete()
    ''

    # Temperature
    img = np.nan_to_num(tifffile.imread('temp.tif'), nan=0) * 0.02
    temp_img = np.transpose(img, (2, 0, 1))
    print(temp_img.shape)
    plt.imshow(temp_img[0]);
    plt.title('MODIS Santa Barbara County: Temperature');
    plt.show()
    # print('SUM OF LAST IMG', temp_img[-1].sum())

    # Moisture
    img = np.nan_to_num(tifffile.imread('moist.tif'), nan=0)
    moist_img = np.transpose(img, (2, 0, 1))
    print(moist_img.shape)
    plt.imshow(moist_img[0]);
    plt.colorbar();
    plt.title('Santa Barbara County: Moisture');
    plt.show()

    # Landcover
    mask_img = np.nan_to_num(tifffile.imread('mask.tif'), nan=0)
    if len(mask_img.shape) == 3:
        mask_img = np.transpose(mask_img, (2, 0, 1))
    if len(mask_img.shape) == 2:
        mask_img = np.reshape(mask_img, (1, mask_img.shape[0], mask_img.shape[1]))
    print('Mask', mask_img.shape)
    plt.imshow(mask_img[0])
    plt.title('Santa Barbara County: Mask');
    plt.show()

    bands = 2
    t_img = []
    for i in range(0, len(temp_img), bands):
        t_img.append(temp_img[i:i + bands])

    m_img = []
    for i in range(0, len(moist_img), bands):
        m_img.append(moist_img[i:i + bands])

    ### Checking Temperature Missing data ###
    ee.Initialize()
    collection = ee.ImageCollection('MODIS/006/MOD11A1').filterDate(str(train_date_start),
                                                                    str(train_date_end + datetime.timedelta(days=1)))
    d_list = np.asarray(ymdList(collection))
    td_list = []
    for i in range(len(d_list)):
        td_list.append(datetime.datetime.strptime(d_list[i], '%Y%m%d'))

    for i in range(len(td_list) - 1):
        if td_list[i] + datetime.timedelta(days=1) != td_list[i + 1]:
            # print('missing image', td_list[i], td_list[i] + datetime.timedelta(days=1))
            td_list.insert(i + 1, td_list[i] + datetime.timedelta(days=1))
            t_img.insert(i + 1, t_img[i])

    t_img = np.asarray(t_img)
    print('Pre Update Temperature:', t_img.shape)

    ### Checking Moisture Missing data ###
    print(str(train_date_start), str(train_date_end + datetime.timedelta(days=1)))
    collection = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture').filterBounds(
        ee.Geometry.Rectangle(-106, 50, -60, 20)).filterDate(str(train_date_start),
                                                             str(train_date_end + datetime.timedelta(days=1)))
    d_list = np.asarray(ymdList(collection))
    md_list = []
    for i in range(len(d_list)):
        md_list.append(datetime.datetime.strptime(d_list[i], '%Y%m%d'))

    for i in range(len(md_list) - 1):
        if md_list[i] + datetime.timedelta(days=3) != md_list[i + 1]:
            print('missing image', md_list[i], md_list[i] + datetime.timedelta(days=3))
            md_list.insert(i + 1, md_list[i] + datetime.timedelta(days=3))
            m_img.insert(i + 1, m_img[i])

    m_img = np.asarray(m_img)
    print('Pre Update Moisture:', m_img.shape)

    mdate = ee.Date(collection.first().get('system:time_start')).format('Y-M-d').getInfo()
    moisture_startdate = datetime.datetime.strptime(mdate, "%Y-%m-%d")
    # print(window_date_start)
    # print(moisture_startdate)
    mdate = ee.Date(collection.sort('system:time_start', False).first().get('system:time_end')).format(
        'Y-M-d').getInfo()
    # print('mdate',mdate)
    moisture_enddate = datetime.datetime.strptime(mdate, "%Y-%m-%d") - datetime.timedelta(days=1)
    # print(window_date_end)
    # print(moisture_enddate)
    # print('end date:', moisture_enddate)
    mm_imgs = []
    for i in range(m_img.shape[0]):
        if i == 0:
            if moisture_startdate.date() == train_date_start:
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
            if moisture_startdate.date() == train_date_start + datetime.timedelta(days=1):
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
            if moisture_startdate.date() == train_date_start + datetime.timedelta(days=2):
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
            continue
        if i == (m_img.shape[0] - 1):
            if moisture_enddate.date() == train_date_end:
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
            if moisture_enddate.date() == train_date_end + datetime.timedelta(days=1):
                mm_imgs.append(m_img[i]);
                mm_imgs.append(m_img[i]);
            if moisture_enddate.date() == train_date_end + datetime.timedelta(days=2):
                mm_imgs.append(m_img[i]);
            break

        mm_imgs.append(m_img[i]);
        mm_imgs.append(m_img[i]);
        mm_imgs.append(m_img[i]);
    m_img = np.array(mm_imgs)
    print('Moisture Updated:', m_img.shape)
    if len(m_img) != len(t_img):
        m_img = m_img[:len(t_img)]
        print('Moisture Updated:', m_img.shape)

    ### Masking Temperature and Moisture ###
    mask_img[mask_img != 12] = 0
    mask_img[mask_img == 12] = 1

    # Masking for every year
    date_list = [];
    x = 0
    while 1:
        d = train_date_start + datetime.timedelta(days=x)
        if d.month == 2 and d.day == 29:
            d = d.replace(day=28)
        if int(d.year) > 2019:
            d = d.replace(year=2019)
        date_list.append(d)
        if train_date_start + datetime.timedelta(days=x) == train_date_end:
            break
        x += 1
    date_list = np.asarray(date_list)

    y = 0
    tmasked_img = t_img.copy()
    mmasked_img = m_img.copy()
    for i in range(len(date_list)):
        for b in range(t_img.shape[1]):
            tmasked_img[i, b] = t_img[i, b] * mask_img[y]
            mmasked_img[i, b] = m_img[i, b] * mask_img[y]
        if i == len(date_list) - 1:
            break
        if date_list[i].year != date_list[i + 1].year:
            y += 1

    # Temperature Histograms
    # bin_seq = np.linspace(12999, 16999, 33)
    bin_seq = np.linspace(250, 350, 33)

    n_imgs = tmasked_img.shape[0]
    bands = tmasked_img.shape[1]
    bins = 32
    temp_hist = np.zeros((n_imgs, bands, bins))

    for i in range(tmasked_img.shape[0]):
        for b in range(tmasked_img.shape[1]):
            density, _ = np.histogram(tmasked_img[i, b], bin_seq, density=False)
            if float(density.sum()) == 0:
                continue
            temp_hist[i, b] = density / float(density.sum())
    for i in range(temp_hist.shape[0]):
        for b in range(temp_hist.shape[1]):
            if np.sum(temp_hist[i, b]) == 0:
                temp_hist[i, b] = temp_hist[i - 1, b]

    # Moisture Histograms
    # bin_seq = bin_seq = np.linspace(1, 149, 33)
    bin_seq = bin_seq = np.linspace(1, 120, 33)

    n_imgs = mmasked_img.shape[0]
    bands = mmasked_img.shape[1]
    bins = 32
    moist_hist = np.zeros((n_imgs, bands, bins))

    for i in range(mmasked_img.shape[0]):
        for b in range(mmasked_img.shape[1]):
            density, _ = np.histogram(mmasked_img[i, b], bin_seq, density=False)
            if float(density.sum()) == 0:
                continue
            moist_hist[i, b] = density / float(density.sum())

    for i in range(moist_hist.shape[0]):
        for b in range(moist_hist.shape[1]):
            if np.sum(moist_hist[i, b]) == 0:
                moist_hist[i, b] = moist_hist[i - 1, b]

    # Combining histograms
    prelagged_hists = np.zeros([moist_hist.shape[0], 4, 32])
    for i in range(prelagged_hists.shape[0]):
        prelagged_hists[i] = np.concatenate((temp_hist[i], moist_hist[i]))
    print('Prelagged hists', prelagged_hists.shape)
    for i in range(len(prelagged_hists)):
        for j in range(len(prelagged_hists[0])):
            if np.sum(prelagged_hists[i, j]) == 0:
                if i - 365 >= 0:
                    prelagged_hists[i, j] = prelagged_hists[i - 365, j]
                else:
                    prelagged_hists[i, j] = prelagged_hists[i - 1, j]

    total = 0
    e = 0
    for i in range(prelagged_hists.shape[0]):
        for b in range(prelagged_hists.shape[1]):
            total += 1
            if np.sum(prelagged_hists[i, b]) == 0:
                e += 1
    # print('Total Histograms:',total)
    print('Missing Histograms:', e)

    return prelagged_hists


def SatImgsTask2(date_start_str, date_end_str, output_type, output_list, county_name, horizon=0):
    if horizon == 0: h = 1
    if horizon == 1: h = 7
    if horizon == 2: h = 7 * 2
    if horizon == 3: h = 7 * 3
    if horizon == 4: h = 7 * 4
    if horizon == 5: h = 7 * 5

    if county_name == 'santa_barbara':
        fips = '06083'

    training_dates = output_list['Date'].dt.date.values

    training_date_start = training_dates[0] - datetime.timedelta(days=140 + h)
    training_date_end = training_dates[-1] - datetime.timedelta(days=h)
    print(training_date_start)
    print(training_date_end)
    prelagged_hists = np.load('prelagged_hists_sm_new_moist.npy')
    data_date_start = datetime.datetime(year=2015, month=4, day=1).date()
    data_date_end = datetime.datetime(year=2019, month=12, day=31).date()
    start_window = (training_date_start - data_date_start).days
    prelagged_hists = prelagged_hists[start_window:]
    print(prelagged_hists.shape)
    end_window = (data_date_end - training_date_end).days
    print(end_window)
    prelagged_hists = prelagged_hists[:-end_window]
    print(prelagged_hists.shape)

    train_prelagged_hists = np.transpose(prelagged_hists, (0, 2, 1))
    llagged_hists = []
    for i in range(len(output_list)):
        llagged_hists.append(train_prelagged_hists[i:i + 140])
    lagged_hists = np.array(llagged_hists)
    print('After lagging', lagged_hists.shape)

    # Reshape
    lagged_hists = np.transpose(lagged_hists, [0, 2, 1, 3])
    lagged_hists = np.reshape(lagged_hists, [lagged_hists.shape[0], -1, lagged_hists.shape[2] * lagged_hists.shape[3]])
    print('Reshaped:', lagged_hists.shape)

    split = int(0.8 * len(lagged_hists))
    hists_train = lagged_hists[:split]
    yields_train = output_list['Yield'].values[:split]
    hists_val = lagged_hists[split:]
    yields_val = output_list['Yield'].values[split:]
    print('Train:', hists_train.shape, yields_train.shape)
    print('Validate:', hists_val.shape, yields_val.shape)

    agm_CNN = 100
    while agm_CNN > 50:
        if output_type == 'yield':
            if h == 1: model = models.load_model('SatelliteModels/best_s2y_sm_1day.hdf5')
            if h == 7: model = models.load_model('SatelliteModels/best_s2y_sm_1week.hdf5')
            if h == 7 * 2: model = models.load_model('SatelliteModels/best_s2y_sm_2week.hdf5')
            if h == 7 * 3: model = models.load_model('SatelliteModels/best_s2y_sm_3week.hdf5')
            if h == 7 * 4: model = models.load_model('SatelliteModels/best_s2y_sm_4week.hdf5')
            if h == 7 * 5: model = models.load_model('SatelliteModels/best_s2y_sm_5week.hdf5')

        if output_type == 'price':
            if h == 1: model = models.load_model('SatelliteModels/best_s2p_sm_1day.hdf5')
            if h == 7: model = models.load_model('SatelliteModels/best_s2p_sm_1week.hdf5')
            if h == 7 * 2: model = models.load_model('SatelliteModels/best_s2p_sm_2week.hdf5')
            if h == 7 * 3: model = models.load_model('SatelliteModels/best_s2p_sm_3week.hdf5')
            if h == 7 * 4: model = models.load_model('SatelliteModels/best_s2p_sm_4week.hdf5')
            if h == 7 * 5: model = models.load_model('SatelliteModels/best_s2p_sm_5week.hdf5')

        for layer in model.layers[:5]:
            layer.trainable = False

        model.pop()
        model.add(layers.Dense(units=512, activation='relu', name='tf1'))
        model.add(layers.Dense(units=1, activation='relu', name='tf2'))

        # print(model.summary())
        start = time.time()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = 'mean_absolute_error'
        model.compile(optimizer=optimizer, loss=loss)
        mcp_save = ModelCheckpoint('SatelliteModels/best_s2y_transfer_learning.hdf5', save_best_only=True,
                                   monitor='val_loss', mode='min')
        history = model.fit(hists_train, yields_train, validation_data=(hists_val, yields_val), epochs=30 \
                            , batch_size=32, callbacks=[mcp_save], verbose=0)

        print('Training time:', time.time() - start, ' seconds')
        model = models.load_model('SatelliteModels/best_s2y_transfer_learning.hdf5')
        pred = model.predict(hists_val).flatten()

        RMSE_CNN = np.sqrt(np.mean((pred - yields_val) ** 2))
        MAE_CNN = np.mean(np.abs(pred - yields_val))
        r2_CNN = r2_score(yields_val, pred)
        agm_CNN = ((RMSE_CNN + MAE_CNN) / 2) * (1 - r2_CNN)
        plt.plot(yields_val, label='True yield');
        plt.plot(pred, label='Predicted yield');
        plt.legend(loc='upper right');
        plt.grid()
        plt.xlabel('Days');
        plt.ylabel('Yield')
        plt.title('Frozen Base Model on Raspberry Data')
        plt.show()
        print(MAE_CNN)
        print(RMSE_CNN)
        print(r2_CNN)
        print(agm_CNN)

    # Forecasting
    date_start = datetime.datetime(int(date_start_str[:4]), int(date_start_str[5:7]), int(date_start_str[8:]), 0, 0)
    date_end = datetime.datetime(int(date_end_str[:4]), int(date_end_str[5:7]), int(date_end_str[8:]), 0, 0)
    window_date_start = date_start - datetime.timedelta(days=140 + h)
    window_date_end = date_end - datetime.timedelta(days=h)
    print(str(window_date_start.date()), str(window_date_end.date()))

    forecast_prelagged_hists = getHists(window_date_start.date(), window_date_end.date(), fips, horizon)

    prelagged_hists = np.transpose(forecast_prelagged_hists, (0, 2, 1))
    llagged_hists = []
    for i in range((date_end + datetime.timedelta(1) - date_start).days):
        llagged_hists.append(prelagged_hists[i:i + 140])
    lagged_hists = np.array(llagged_hists)
    print('After lagging', lagged_hists.shape)

    # Reshape
    lagged_hists = np.transpose(lagged_hists, [0, 2, 1, 3])
    lagged_hists = np.reshape(lagged_hists, [lagged_hists.shape[0], -1, lagged_hists.shape[2] * lagged_hists.shape[3]])
    print('Reshaped:', lagged_hists.shape)

    ### Obtaining forecasts ###
    if output_type == 'yield':
        model = models.load_model('SatelliteModels/best_s2y_transfer_learning.hdf5')

    # print(model.summary())
    preds = model.predict(lagged_hists)
    preds = ['%.2f' % elem[0] for elem in preds]
    return preds


'''
d_start = '2019-05-01'
d_end = '2019-07-01'


rasp_list = pd.read_excel('raspberries_sm.xlsx')
y = SatImgsTask2(d_start, d_end, output_type='yield', output_list=rasp_list, county_name='santa_barbara', horizon=5)

#%%
true_y = pd.read_excel('raspberries_sm.xlsx')['Yield'].values[1216:1278]
fit1 = ExponentialSmoothing(y, seasonal_periods=12).fit()
y_exp = fit1.fittedvalues

plt.plot(true_y, label='true')
plt.plot(y, label='predicted')
plt.plot(y_exp, label='smoothing')
plt.legend()
plt.show()
'''