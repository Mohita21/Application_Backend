from Station_Training import Att_CNN_LSTM as acl1, Data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt

n=12
d = pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Data/Ifeanyi_soil_weather_yield_price_5W.xlsx")
df=pd.read_excel("/Users/mohita/Documents/GitHub/Flask_app/Data/LatestMoistTemp.xlsx")
yld = d["Yield"]
#p = [abs(i) * 1 for i in d["Price"]]
p = d["Price"]
#print(np.isnan(p).any())
x = df["Soil Temperature"]
print("ifeanyi",len(yld))
print("Latest",len(x))
y=p
plt.plot(y)
plt.show()
path= "../TrainingWeights/weights_200_price_1D.hdf5"
scl="sclrWeights/sclr_price_1D.pkl"
pca="pcaWeights/pca_price_1D.pkl"
#x_tl=d["Soil Temperature"][-300:]
#y_tl=p[-300:]

x_train,y_train,x_test,y_test=dp.read_data_and_preprocessing(x,y,n,scl,pca)
model_lime_cnn_lstm_att = acl1.model(n)
model_lime_cnn_lstm_att,filepath_lime_cnn_lstm_att,history_lime_cnn_lstm_att=acl1.train(model_lime_cnn_lstm_att,x_train,y_train,x_test,y_test,path)
acl1.test(model_lime_cnn_lstm_att,filepath_lime_cnn_lstm_att,history_lime_cnn_lstm_att,x_test,y_test)
acl1.test_train_data(model_lime_cnn_lstm_att, "../TrainingWeights/weights_200_price_1D.hdf5", x_train, y_train)
#x_tl_train,y_tl_train,x_tl_test,y_tl_test=dp.read_data_and_preprocessing(x_tl,y_tl,n)

#final_model,updated_path, history=acl1.transfer_learning(n,filepath_lime_cnn_lstm_att,x_tl_train,y_tl_train,x_tl_test,y_tl_test)
#acl1.test(final_model,updated_path, history,x_test,y_test)

#acl2.train_and_test(x_train,y_train,x_test,y_test)
#pickle.dump(model_lime_cnn_lstm_att, open('model2','wb'))

#print(x_test.shape)

