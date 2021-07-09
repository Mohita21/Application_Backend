from datetime import datetime, timedelta
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import logging
import Att_CNN_LSTM as acl1
import matplotlib.pyplot as plt
import Data_preprocessing as dp
import pandas as pd
import numpy as np
#import Att_Conv_LSTM1 as acl2
import pickle
from flask import Flask, render_template,request
import MoisTempDateRange as mt
#from sklearn.externals import joblib
import joblib
import Satellite as sat
from flask import send_file
import SimilarityCheck as simcheck
import Detailed_Analysis as analysis
import Transfer_learning as TL
import Transfer_Learning_Sat as TL_sat
import Transfer_Learning_Sat2 as TL_sat2
import Saad_Analysis as SA
import Saad_chunks as SC

n=12
model_lime_cnn_lstm_att = acl1.model(n)


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

def print_factors(x):
    factors=[]
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors

def forecast(weight_path,sclr_path,pca_path,test):
    model_lime_cnn_lstm_att.load_weights(weight_path,)
    scaler = joblib.load(sclr_path)
    test = scaler.transform(test)
    my_model = joblib.load(pca_path)
    test = my_model.transform(test)
    test = test.reshape(test.shape[0], test.shape[1], 1)
    prediction = model_lime_cnn_lstm_att.predict(test)
    prediction = prediction.tolist()
    fig = plt.figure()
    plt.plot(prediction, label="pred")
    plt.legend()
    fig.savefig("Prediction.jpg")
    plt.show()
    prediction = ['%.2f' % elem[0] for elem in prediction]
    return prediction



def date_span(sd,ed):
    dat = []
    sd = datetime.strptime(sd, '%Y-%m-%d')
    ed = datetime.strptime(ed, '%Y-%m-%d')
    delta = ed - sd  # as timedelta
    for i in range(delta.days + 1):
        day = sd + timedelta(days=i)
        dat.append(day)
    return dat

api_v2_cors_config = {
    "origins": "http://localhost:4200",
    "methods": ["OPTIONS", "GET", "POST"],
    "allow_headers": ["Authorization", "Content-Type"]
}

app = Flask(__name__)
# default page of our web-app

CORS(app, resources={
    # r"/api/*": api_v1_cors_config,
    r'/predict/': api_v2_cors_config
})
logging.getLogger('flask_cors').level = logging.DEBUG



# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.get_json()
    sd=data['sd']
    ed=data['ed']
    ft= data['ft']
    wa=data['wa']
    mtt=data['mt']
    dat=date_span(sd,ed)
    if ft=='Yield':
        if wa=='0.5':
            test = mt.date_range(sd, ed,0.5)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=0)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_1D.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_1D.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=0)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_1D.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_1D.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='1':
            test = mt.date_range(sd, ed, 1)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=1)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_1W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_1W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=1)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_1W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_1W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='2':
            test = mt.date_range(sd, ed, 2)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=2)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_2W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_2W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=2)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_2W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_2W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='3':
            test = mt.date_range(sd, ed, 3)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=3)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_3W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_3W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=3)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_3W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_3W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='4':
            test = mt.date_range(sd, ed, 4)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=4)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_4W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_4W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=4)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_4W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_4W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='5':
            test = mt.date_range(sd, ed, 5)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='yield', horizon=5)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_5W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_5W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='yield', horizon=5)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_yield_5W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_yield_5W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
    if ft=='Price':
        if wa=="0.5":
            test = mt.date_range(sd, ed, 0.5)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=0)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction = forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_1D.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_1D.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=0)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_1D.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_1D.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="1":
            test = mt.date_range(sd, ed, 1)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=1)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction = forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_1W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_1W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=1)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_1W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_1W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="2":
            test = mt.date_range(sd, ed, 2)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=2)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction = forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_2W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_2W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=2)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_2W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_2W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="3":
            test = mt.date_range(sd, ed, 3)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=3)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction = forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_3W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_3W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=3)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_3W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_3W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="4":
            test = mt.date_range(sd, ed, 4)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=4)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction = forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_4W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_4W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=4)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_4W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_4W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="5":
            test = mt.date_range(sd, ed, 5)
            if mtt=='Satellite':
                preds=sat.PredictSatImgs(sd,ed,output_type='price', horizon=5)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                prediction= forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_5W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_5W.pkl", test)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                preds = sat.PredictSatImgs(sd, ed, output_type='price', horizon=5)
                prediction=forecast("/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5", "/Users/mohita/Documents/GitHub/Flask_app/sclrWeights/sclr_price_5W.pkl", "/Users/mohita/Documents/GitHub/Flask_app/pcaWeights/pca_price_5W.pkl", test)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
    #filename= "Prediction.jpg"
    #return send_file(filename, mimetype='image/gif')

@app.route('/ChunkImputation', methods=['POST'])
@cross_origin()
def ChunkImputation():
    data = request.get_json()
    d1=data[0]
    d2=data[1]
    val1=[]
    val2=[]
    for  value in d1['result']:
        key= list(value.keys())[0]
        v= value.get(key)
        if v =='\"\"' or v == "" or not v:
            val1.append(np.nan)
        else:
            val1.append(float(v))
    for  value in d2['result']:
        key= list(value.keys())[0]
        v= value.get(key)
        if v =='\"\"' or v == "" or not v:
            val2.append(np.nan)
        else:
            val2.append(float(v))
    print(val1)
    print(val2)
    cf=SC.merger(val2,val1)
    SC.Imputer(cf)
    dff=[]
    df=pd.read_excel("Transfer Learning Results.xls")
    df=df["Imputed"]
    for i in range(len(df)):
        dff.append([df[i]])
    return jsonify({'message': dff})




@app.route('/TS_analysis', methods=['POST'])
@cross_origin()
def TS_analysis():
    data = request.get_json()

    val = []
    for value in data['result']:
        key = list(value.keys())[0]
        v = value.get(key)
        if v == '\"\"' or v == "" or not v or v == "nan":
            val.append(-1)
        else:
            val.append(float(v))
    ts_type, chunk_flag = SA.Analyzer(val)
    print(chunk_flag)

    return jsonify({ "type": ts_type, "chunk": chunk_flag})


@app.route('/univariate_impute', methods=['POST'])
@cross_origin()
def univariate_impute():
    data = request.get_json()
    #print(data['result'])
    print("Data",data)

    val=[]
    for  value in data['result']:
        key= list(value.keys())[0]
        v= value.get(key)
        if v =='\"\"' or v == "" or not v:
            val.append(-1)
        else:
            val.append(float(v))
    for i in range(len(val)):
        if  np.isnan(val[i]):
            val[i]=-1
    #ts_type, chunk_flag = SA.Analyzer(val)

    print(val)
    print(type(val[1]))
    df= pd.DataFrame(val)
    plt.plot(df)
    plt.show()
    df.to_csv('miss_data.csv',index=None,header=None)
    row = pd.read_csv('miss_data.csv', sep='\t', header=None).shape[0]
    steps = 34
    batch = 21
    prime_batch = prime_numbers(batch)
    num_data = row - steps
    difference = []
    new_prime = []

    while True:
        if len(prime_numbers(row - steps)) == 1:
            steps = steps - 1
            print(steps)

        if len(prime_numbers(row - steps)) < 4:
            steps = steps - 1
            print(steps)

        for prb in prime_batch:
            if prb in prime_numbers(row - steps):
                new_prime.append(prb)

        if len(new_prime) == len(prime_batch):
            batch = np.prod(new_prime)
            break

        else:
            batch = min(print_factors(row - steps), key=lambda x: abs(x - batch))
            break

    print('batch', batch)
    print('steps', steps)
    FLAGS = None
    #exec(open('Residual_GRU.py').read())
    run = f'python Residual_GRU.py {steps} {batch}'
    os.system(run)
    #os.system('python Residual_GRU.py {steps} {batch}')
    df=pd.read_excel("imputed data Residual.xlsx")
    #df=df.values
    plt.plot(df)
    plt.show()
    df=df.values.tolist()
    print(df)
    for i in range(len(df)):
        #print(df[i])
        if df[i][0]<0:
            df[i][0]=0


    return jsonify({'message':df})

@app.route('/SimilarityCheck', methods=['POST'])
@cross_origin()
def SimilarityCheck():
    data = request.get_json()
    #print(data)
    #print(len(data))

    form=data[0]
    mt= form["method"]
    print("method",mt)
    if mt=="1":
        a = data[1][2]
        b = data[1][3]
        a2= data[2][2]
        b2= data[2][3]
        a = [float(item) for item in a]
        b = [float(item) for item in b]
        a2 = [float(item) for item in a2]
        b2 = [float(item) for item in b2]
        c1 = pd.DataFrame({'Soil Temperature': a})
        c1["Soil Moisture"] = b
        c2 = pd.DataFrame({'Soil Temperature': a2})
        c2["Soil Moisture"] = b2
        print(c1)
        print(c2)
        print(len(c1))
        print(len(c2))
        StartDate = '2015-04-01'
        per, bin = simcheck.Similarity_Input(StartDate,c1, c2)
        #analysis.DeResults_Similarity_Input(StartDate,c1, c2)
        print(per, bin)
        return jsonify({'message': per, 'binary_value': bin})
    if mt=="2":
        y1 = data[1][1]
        y2 = data[2][1]
        d1 = data[1][0]
        d2 = data[2][0]
        y1 = [float(item) for item in y1]
        y2 = [float(item) for item in y2]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        print(y1)
        print(y2)
        percentage,binary_check=simcheck.Similarity_Output(y1,y2,dates)
        #analysis.DeResults_Similarity_Output(y1, y2, dates)
        #analysis.DeResults_Similarity_Output(y1,y2,dates)
        print(percentage,binary_check)
        return jsonify({'message': percentage, 'binary_value': binary_check})
    if mt=="3":
        a = data[1][2]
        b = data[1][3]
        a2= data[2][2]
        b2= data[2][3]
        a = [float(item) for item in a]
        b = [float(item) for item in b]
        a2 = [float(item) for item in a2]
        b2 = [float(item) for item in b2]
        c1 = pd.DataFrame({'Soil Temperature': a})
        c1["Soil Moisture"] = b
        c2 = pd.DataFrame({'Soil Temperature': a2})
        c2["Soil Moisture"] = b2
        y1= data[1][1]
        y2= data[2][1]
        d1= data[1][0]
        d2= data[2][0]
        y1 = [float(item) for item in y1]
        y2 = [float(item) for item in y2]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        #percentage,binary_check=simcheck.Similarity_Input_Output(c1, c2, y1,y2,dates)
        #analysis.Similarity_Input_Output(c1, c2, y1,y2,dates)
        #analysis.DeResults_Similarity_Output(y1,y2,dates)
        #print(percentage,binary_check)
        return jsonify({'message': 78, 'binary_value': 1})
    #return jsonify({'message': data})

@app.route('/SimilarOutputPrice', methods=['POST'])
@cross_origin()
def SimilarOutputPrice():
    data = request.get_json()
    if len(data) ==2:
        y11 = pd.read_csv("Strawberry_Price.csv")
        d1 = y11["Date"]
        y1 = y11["Strawberry"]
        y1 = y1.tolist()
        y2 = data[1]
        y2 = [float(item) for item in y2]
        d2=data[0]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        percentage, binary_check = simcheck.Similarity_Output(y1, y2, dates)
        print(percentage,binary_check)
        return jsonify({'message': percentage, 'binary_value': binary_check})
    if len(data)>2:
        y11 = pd.read_csv("Strawberry_Price.csv")
        d1 = y11["Date"]
        y1 = y11["Strawberry"]
        y1 = y1.tolist()
        y2 = data[1]
        y2 = [float(item) for item in y2]
        d2=data[0]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        percentage, binary_check = simcheck.Similarity_Output(y1, y2, dates)
        print(percentage,binary_check)
        c1 = pd.read_excel("County_similarity.xlsx")
        a = data[2]
        b = data[3]
        #print(a)
        a = [float(item) for item in a]
        b = [float(item) for item in b]
        c2 = pd.DataFrame({'Soil Temperature': a})
        c2["Soil Moisture"] = b
        print(c1)
        print(c2)
        print(len(c1))
        print(len(c2))

        StartDate = '2015-04-01'
        per, bin = simcheck.Similarity_Input(StartDate,c1, c2)
        print(per, bin)
        return jsonify({'message': percentage, 'binary_value': binary_check,'message2': per,'binary_value_ip': bin})

@app.route('/SimilarOutput', methods=['POST'])
@cross_origin()
def SimilarOutput():
    data = request.get_json()
    if len(data) ==2:
        y11 = pd.read_csv("Strawberry_Yield.csv")
        d1 = y11["Date"]
        y1 = y11["Strawberry"]
        y1 = y1.tolist()
        y2 = data[1]
        y2 = [float(item) for item in y2]
        d2=data[0]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        print(len(y1))
        print(len(dates))
        print("&&&&&&&&&&&",dates)
        percentage, binary_check = simcheck.Similarity_Output(y1, y2, dates)
        print(percentage,binary_check)
        return jsonify({'message': percentage, 'binary_value': binary_check})
    if len(data)>2:
        y11 = pd.read_csv("Strawberry_Yield.csv")
        d1 = y11["Date"]
        y1 = y11["Strawberry"]
        y1 = y1.tolist()
        y2 = data[1]
        y2 = [float(item) for item in y2]
        d2=data[0]
        y1, y2, dates = simcheck.Common_Yield_Window(d1, d2, y1, y2)
        percentage, binary_check = simcheck.Similarity_Output(y1, y2, dates)
        print(percentage,binary_check)
        c1 = pd.read_excel("County_similarity.xlsx")
        a = data[2]
        b = data[3]
        #print(a)
        a = [float(item) for item in a]
        b = [float(item) for item in b]
        c2 = pd.DataFrame({'Soil Temperature': a})
        c2["Soil Moisture"] = b
        print(c1)
        print(c2)
        print(len(c1))
        print(len(c2))

        StartDate = '2015-04-01'
        per, bin = simcheck.Similarity_Input(StartDate,c1, c2)
        print(per, bin)
        return jsonify({'message': percentage, 'binary_value': binary_check,'message2': per,'binary_value_ip': bin})

@app.route('/TransferLearningOp', methods=['POST'])
@cross_origin()
def TransferLearningOp():
    data = request.get_json()
    form =data[0]
    new_file = data[1]
    d1=new_file[0]
    y1 = new_file[1]
    y1 = [float(item) for item in y1]
    sd=form['sd']
    ed=form['ed']
    ft= form['ft']
    wa=form['wa']
    mtt=form['mt']
    dat=date_span(sd,ed)
    if ft=='Yield':
        if wa=='0.5':
            test = mt.date_range(sd, ed,0.5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame=d_frame.iloc[1532:,:]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=0)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5"
                prediction=TL.TL_out(model_lime_cnn_lstm_att,weight_path,test,y1,n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame=d_frame.iloc[1532:,:]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=0)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5"
                prediction=TL.TL_out(model_lime_cnn_lstm_att,weight_path,test,y1,n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='1':
            test = mt.date_range(sd, ed, 1)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=1)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=1)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='2':
            test = mt.date_range(sd, ed, 2)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=2)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=2)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='3':
            test = mt.date_range(sd, ed, 3)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=3)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=3)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='4':
            test = mt.date_range(sd, ed, 4)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=4)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=4)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='5':
            test = mt.date_range(sd, ed, 5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=5)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara', horizon=5)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
    if ft=='Price':
        if wa=="0.5":
            test = mt.date_range(sd, ed, 0.5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=0)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=0)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="1":
            test = mt.date_range(sd, ed, 1)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=1)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=1)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="2":
            test = mt.date_range(sd, ed, 2)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=2)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=2)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="3":
            test = mt.date_range(sd, ed, 3)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=3)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=3)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="4":
            test = mt.date_range(sd, ed, 4)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=4)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=4)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=="5":
            test = mt.date_range(sd, ed, 5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=5)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                d_frame = d_frame.iloc[1532:, :]
                print(d_frame)
                preds=TL_sat.SatImgsTask2(sd, ed, output_type='price', output_list=d_frame, county_name='santa_barbara', horizon=5)
                print(preds)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5"
                prediction = TL.TL_out(model_lime_cnn_lstm_att, weight_path, test, y1, n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
    return jsonify({'result': y1, 'dates': d1})


@app.route('/TransferLearningIn', methods=['POST'])
@cross_origin()
def TransferLearningIn():
    data = request.get_json()
    form =data[0]
    new_file = data[1]
    d1=new_file[0]
    y1 = new_file[1]
    if len(new_file) >2:
        a = new_file[2]
        b = new_file[3]
        a = [float(item) for item in a]
        b = [float(item) for item in b]
        c2 = pd.DataFrame({'Soil Temperature': a})
        c2["Soil Moisture"] = b
    y1 = [float(item) for item in y1]
    sd=form['sd']
    ed=form['ed']
    ft= form['ft']
    wa=form['wa']
    mtt=form['mt']
    dat=date_span(sd,ed)
    fips=form['fips']
    if ft=='Yield':
        if wa=='0.5':
            test = mt.date_range(sd, ed,0.5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame=d_frame.iloc[1532:,:]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=0)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print("****************************", d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                print("****************************",d_frame)
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=0)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1D.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='1':
            test = mt.date_range(sd, ed, 1)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=1)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=1)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_1W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='2':
            test = mt.date_range(sd, ed, 2)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=2)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=2)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_2W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='3':
            test = mt.date_range(sd, ed, 3)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=3)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=3)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_3W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='4':
            test = mt.date_range(sd, ed, 4)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=4)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=4)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_4W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
        if wa=='5':
            test = mt.date_range(sd, ed, 5)
            if mtt=='Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=5)
                print(preds)
                return jsonify({'result':preds, 'dates':dat})
            if mtt=='Station':
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                return jsonify({'result':prediction, 'dates':dat})
            if mtt=='Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"]=y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds=TL_sat2.SatImgsTask3(sd, ed, output_type='yield', output_list=d_frame, county_name='santa_barbara',fips=fips, horizon=5)
                weight_path="/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_yield_5W.hdf5"
                prediction=TL.TL_in(model_lime_cnn_lstm_att,weight_path,test,y1,c2.iloc[:,:1],n)
                final_pred=[float(preds[i])+float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result':pred_final, 'dates':dat})
    if ft=='Price':
        if wa=="0.5":
            test = mt.date_range(sd, ed, 0.5)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=0)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=0)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1D.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})
        if wa == '1':
            test = mt.date_range(sd, ed, 1)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=1)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=1)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_1W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})
        if wa == '2':
            test = mt.date_range(sd, ed, 2)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=2)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=2)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_2W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})
        if wa == '3':
            test = mt.date_range(sd, ed, 3)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=3)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=3)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_3W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})
        if wa == '4':
            test = mt.date_range(sd, ed, 4)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=4)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1,c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=4)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_4W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1, c2.iloc[:,:1],n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})
        if wa == '5':
            test = mt.date_range(sd, ed, 5)
            if mtt == 'Satellite':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=5)
                print(preds)
                return jsonify({'result': preds, 'dates': dat})
            if mtt == 'Station':
                print("Type1",type(c2.iloc[:,:1]))
                print(c2.iloc[:,:1])
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test,y1, c2.iloc[:,:1], n)
                return jsonify({'result': prediction, 'dates': dat})
            if mtt == 'Combined':
                d_frame = pd.DataFrame({'Date': pd.DatetimeIndex(d1)})
                d_frame["Yield"] = y1
                print(d_frame)
                #d_frame = d_frame.iloc[1532:, :]
                preds = TL_sat2.SatImgsTask3(sd, ed, output_type='price', output_list=d_frame,
                                             county_name='santa_barbara', fips=fips, horizon=5)
                weight_path = "/Users/mohita/Documents/GitHub/Flask_app/TrainingWeights/weights_200_price_5W.hdf5"
                prediction = TL.TL_in(model_lime_cnn_lstm_att, weight_path, test, y1, c2.iloc[:,:1],n)
                final_pred = [float(preds[i]) + float(prediction[i]) for i in range(len(preds))]
                pred_final = [x / 2 for x in final_pred]
                return jsonify({'result': pred_final, 'dates': dat})



if __name__ == "__main__":

    app.run(debug=True)
