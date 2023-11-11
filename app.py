import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
import streamlit as st
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mplfinance as mpf

sp = pd.read_csv('SP500.csv')
sp = sp[['Symbol','Name']]
sym = list(sp['Symbol'])
names = list(sp['Name'])
dct = {}
for i in range(len(names)):
    dct[names[i]] = sym[i]


def fetch_real_time_data(symbol):
    data = yf.download(symbol, interval="1d", progress=False)
    return data.to_csv("{}.csv".format(symbol))

def create_dataset(dataset,time_step=1):
    dataX=[]
    dataY=[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        b = dataset[i+time_step,0]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX),np.array(dataY)


def calc_output(temp_input,x_input,model):
    lst_output = []
    steps = 100
    i=0
    while i<30:
        if(len(temp_input)>100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1,steps,1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input =temp_input[1:]
            lst_output.extend(yhat.tolist())
        else:
            x_input = x_input.reshape((1,steps,1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
        i = i+1
    return lst_output


def lstm(symbol):
    df = pd.read_csv('{}.csv'.format(symbol))
    df = df.tail(1000)
    df1 = df.reset_index()['Close']
    df2 = df.reset_index()['Low']
    df3 = df.reset_index()['High']
    df4 = df.reset_index()['Open']
    
    scalar = MinMaxScaler(feature_range=(0,1))
    df1=scalar.fit_transform(np.array(df1).reshape(-1,1))
    df2=scalar.fit_transform(np.array(df2).reshape(-1,1))
    df3=scalar.fit_transform(np.array(df3).reshape(-1,1))
    df4=scalar.fit_transform(np.array(df4).reshape(-1,1))
    train_size = int(len(df1)*0.70)
    test_size = len(df1)-train_size
    
    train_data_d1,test_data_d1=df1[0:train_size,:],df1[train_size:len(df1),:1]
    train_data_d2,test_data_d2=df2[0:train_size,:],df2[train_size:len(df1),:1]
    train_data_d3,test_data_d3=df3[0:train_size,:],df3[train_size:len(df1),:1]
    train_data_d4,test_data_d4=df4[0:train_size,:],df4[train_size:len(df1),:1]
    

    time_step=100
    x_train_d1,y_train_d1 = create_dataset(train_data_d1,time_step)
    x_test_d1,y_test_d1 = create_dataset(test_data_d1,time_step)
    
    x_train_d2,y_train_d2 = create_dataset(train_data_d2,time_step)
    x_test_d2,y_test_d2 = create_dataset(test_data_d2,time_step)
    
    x_train_d3,y_train_d3 = create_dataset(train_data_d3,time_step)
    x_test_d3,y_test_d3 = create_dataset(test_data_d3,time_step)
    
    x_train_d4,y_train_d4 = create_dataset(train_data_d4,time_step)
    x_test_d4,y_test_d4 = create_dataset(test_data_d4,time_step)
    
    
    x_train_d1 =x_train_d1.reshape(x_train_d1.shape[0],x_train_d1.shape[1],1)
    x_test_d1 =x_test_d1.reshape(x_test_d1.shape[0],x_test_d1.shape[1],1)
    x_train_d2 =x_train_d2.reshape(x_train_d2.shape[0],x_train_d2.shape[1],1)
    x_test_d2 =x_test_d2.reshape(x_test_d2.shape[0],x_test_d2.shape[1],1)
    x_train_d3 =x_train_d3.reshape(x_train_d3.shape[0],x_train_d3.shape[1],1)
    x_test_d3 =x_test_d3.reshape(x_test_d3.shape[0],x_test_d3.shape[1],1)
    x_train_d4 =x_train_d4.reshape(x_train_d4.shape[0],x_train_d4.shape[1],1)
    x_test_d4 =x_test_d4.reshape(x_test_d4.shape[0],x_test_d4.shape[1],1)
    
    model1 = Sequential()
    model1.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model1.add(LSTM(50,return_sequences=True))
    model1.add(LSTM(50))
    model1.add(Dense(1))
    model1.compile(loss="mean_squared_error",optimizer="adam")
    model1.fit(x_train_d1,y_train_d1,validation_data=(x_test_d1,y_test_d1),epochs=1,batch_size=64,verbose=1)
    
    model2 = Sequential()
    model2.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model2.add(LSTM(50,return_sequences=True))
    model2.add(LSTM(50))
    model2.add(Dense(1))
    model2.compile(loss="mean_squared_error",optimizer="adam")
    model2.fit(x_train_d2,y_train_d2,validation_data=(x_test_d2,y_test_d2),epochs=1,batch_size=64,verbose=1)
    
    model3 = Sequential()
    model3.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model3.add(LSTM(50,return_sequences=True))
    model3.add(LSTM(50))
    model3.add(Dense(1))
    model3.compile(loss="mean_squared_error",optimizer="adam")
    model3.fit(x_train_d3,y_train_d3,validation_data=(x_test_d3,y_test_d3),epochs=1,batch_size=64,verbose=1)
    
    model4 = Sequential()
    model4.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model4.add(LSTM(50,return_sequences=True))
    model4.add(LSTM(50))
    model4.add(Dense(1))
    model4.compile(loss="mean_squared_error",optimizer="adam")
    model4.fit(x_train_d4,y_train_d4,validation_data=(x_test_d4,y_test_d4),epochs=1,batch_size=64,verbose=1)
    
    train_predict_d1 = model1.predict(x_train_d1)
    test_predict_d1 = model1.predict(x_test_d1)
    
    train_predict_d2 = model2.predict(x_train_d2)
    test_predict_d2 = model2.predict(x_test_d2)
    
    train_predict_d3 = model3.predict(x_train_d3)
    test_predict_d3 = model3.predict(x_test_d3)
    
    train_predict_d4 = model4.predict(x_train_d4)
    test_predict_d4 = model4.predict(x_test_d4)

    train_predict_d1 = scalar.inverse_transform(train_predict_d1)
    test_predict_d1 = scalar.inverse_transform(test_predict_d1)
    train_predict_d2 = scalar.inverse_transform(train_predict_d2)
    test_predict_d2 = scalar.inverse_transform(test_predict_d2)
    train_predict_d3 = scalar.inverse_transform(train_predict_d3)
    test_predict_d3 = scalar.inverse_transform(test_predict_d3)
    train_predict_d4 = scalar.inverse_transform(train_predict_d4)
    test_predict_d4 = scalar.inverse_transform(test_predict_d4)
    
    x_input_d1 = test_data_d1[(len(test_data_d1)-100-1):].reshape(1,-1)
    temp_input_d1 = list(x_input_d1)
    temp_input_d1 = temp_input_d1[0].tolist()    
    x_input_d2 = test_data_d2[(len(test_data_d1)-100-1):].reshape(1,-1)
    temp_input_d2 = list(x_input_d2)
    temp_input_d2 = temp_input_d2[0].tolist()    
    x_input_d3 = test_data_d3[(len(test_data_d1)-100-1):].reshape(1,-1)
    temp_input_d3 = list(x_input_d3)
    temp_input_d3 = temp_input_d3[0].tolist()    
    x_input_d4 = test_data_d4[(len(test_data_d1)-100-1):].reshape(1,-1)
    temp_input_d4 = list(x_input_d4)
    temp_input_d4 = temp_input_d4[0].tolist()    
    
    
    lst_output_d1 = calc_output(temp_input_d1,x_input_d1,model1)
    lst_output_d2 = calc_output(temp_input_d2,x_input_d2,model2)
    lst_output_d3 = calc_output(temp_input_d3,x_input_d3,model3)
    lst_output_d4 = calc_output(temp_input_d4,x_input_d4,model4)
    lst_output_d1 = scalar.inverse_transform(lst_output_d1)
    lst_output_d2 = scalar.inverse_transform(lst_output_d2)
    lst_output_d3 = scalar.inverse_transform(lst_output_d3)
    lst_output_d4 = scalar.inverse_transform(lst_output_d4)
    print(lst_output_d1)
    print(lst_output_d2)
    print(lst_output_d3)
    print(lst_output_d4)
    mse = math.sqrt(mean_squared_error(y_test_d1, test_predict_d1))
    r2 = r2_score(y_test_d1, test_predict_d1)
    print(mse)
    print(r2)
    return lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4,mse,r2


def make_plot(symbol,lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4):
    close=[]
    open_=[]
    high=[]
    low=[]
    for x in lst_output_d1:
        close.append(x[0])
    for x in lst_output_d2:
        low.append(x[0])
    for x in lst_output_d3:
        high.append(x[0])
    for x in lst_output_d4:
        open_.append(x[0])
    df_df = pd.DataFrame({
        "close":close,
        "open":open_,
        "high":high,
        "low":low
    })
    dates = []
    for i in range(1,31):
        date = pd.to_datetime('today')+pd.Timedelta(i,unit='D')
        dates.append(date)

    df_df['date'] = dates
    df_df['date'] = df_df['date'].apply(lambda x : pd.to_datetime(str(x)))
    df_df['date'] = df_df['date'].dt.date
    df_df['date'] = pd.to_datetime(df_df['date'], format='%Y-%m-%d')
    df_df.set_index("date",inplace=True)
    # print("Prediction for next 30 days: ")
    # print(df_df)
    st.write("Prediction of next 30 days : ")
    st.write(df_df)
    df_df.reset_index(inplace=True)
    tmp_df = df_df
    tmp_df.set_index("date",inplace=True)
    plt.plot(dates,lst_output_d1,label='Close')
    plt.plot(dates,lst_output_d2,label='high')
    plt.plot(dates,lst_output_d3,label='low')
    plt.plot(dates,lst_output_d4,label='open')
    plt.legend()
    plt.savefig("plot0_{}.png".format(symbol))
    mpf.plot(df_df,style='yahoo',type='candle')
    # plt.show()
    plt.savefig('plot_{}.png'.format(symbol))
    st.image('plot0_{}.png'.format(symbol))
    st.image('plot_{}.png'.format(symbol))


def gbm(dataset):
    df = pd.read_csv(dataset)
    df['Date'] = pd.to_datetime(df['Date'])
    df2 = df.set_index('Date')
    data = list(df2['Close'])

    x_train = data[:-100]
    x_test = data[-100:]

    dtrain = xgb.DMatrix(np.array(range(len(x_train))).reshape(-1, 1), label=x_train)
    dtest = xgb.DMatrix(np.array(range(len(x_train), len(x_train) + len(x_test))).reshape(-1, 1), label=x_test)

    params = {
        "objective": "reg:squarederror", 
        "max_depth": 3,                  
        "learning_rate": 0.1,
        "n_estimators": 100,              
    }

    model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])

    y_pred = model.predict(dtest)
    
    mse = mean_squared_error(x_test, y_pred)
    r2 = r2_score(x_test, y_pred)

    return mse, r2

def arima(dataset):
    # Read the dataset from a CSV file
    df = pd.read_csv(dataset)
    df = df.tail(1000)
    df['Date'] = pd.to_datetime(df['Date'])
    df2 = df.set_index('Date')
    data = list(df2['Close'])

    # Fit an ARIMA model using auto_arima to determine the order
    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True)
    order = stepwise_fit.get_params()['order']

    # Split data into training and testing sets
    x_train = data[:-100]
    x_test = data[-100:]

    # Fit the ARIMA model
    model = sm.tsa.ARIMA(data, order=order)
    model = model.fit()
    print(model.summary())

    # Make predictions for the test set
    start = len(x_train)
    end = len(x_train) + len(x_test) - 1
    pred = model.predict(start=start, end=end, typ='levels')

    # Calculate RMSE and R2 score
    mse = mean_squared_error(x_test, pred)
    r2 = r2_score(x_test, pred)

    return mse, r2

def randomForest(dataset):
    df = pd.read_csv(dataset)
    df = df.tail(1000)
    df1 = df.reset_index()['Close']
    scalar = MinMaxScaler(feature_range=(0,1))
    df1=scalar.fit_transform(np.array(df1).reshape(-1,1))
    train_size = int(len(df1)*0.70)
    test_size = len(df1)-train_size
    train_data_d1,test_data_d1=df1[0:train_size,:],df1[train_size:len(df1),:1]
    time_step=100
    x_train_d1,y_train_d1 = create_dataset(train_data_d1,time_step)
    x_test_d1,y_test_d1 = create_dataset(test_data_d1,time_step)
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(x_train_d1, y_train_d1)    
    RandomForestRegressor(random_state=0)
    train_predict=regressor.predict(x_train_d1)
    test_predict=regressor.predict(x_test_d1)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
    train_predict = scalar.inverse_transform(train_predict)
    test_predict = scalar.inverse_transform(test_predict)
    y_train_d1 = scalar.inverse_transform(y_train_d1.reshape(-1,1)) 
    y_test_d1 = scalar.inverse_transform(y_test_d1.reshape(-1,1))     
    
    mse = mean_squared_error(y_test_d1, test_predict)
    r2 = r2_score(y_test_d1,test_predict)
    return mse,r2


def MLP(dataset):
    df = pd.read_csv(dataset)
    df = df.tail(1000)
    df["Diff"] = df.Close.diff()
    df["SMA_2"] = df.Close.rolling(2).mean()
    df["Force_Index"] = df["Close"] * df["Volume"]
    df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
    df = df.drop(
       ["Date","Open", "High", "Low", "Close", "Volume", "Diff", "Adj Close"],
       axis=1,
    ).dropna()
    X = df.drop(["y"], axis=1).values
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.2,
       shuffle=False,
    )
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, shuffle=False))
    clf.fit(
       X_train,
       y_train,
    )
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return mse,r2


def MA_Strategy(stock,MAF,MAS):
    data = pd.read_csv(stock)
    data = data.tail(1000)
    data['Log Returns'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    data['MASlow'] = data['Adj Close'].rolling(MAS).mean()
    data['MAFast'] = data['Adj Close'].rolling(MAF).mean()
    data.dropna(inplace=True)
    data['Signal'] = np.where(data['MAFast']>data['MASlow'],1,-1)
    data.dropna(inplace=True)
    data['Strategy Log Returns'] = data['Log Returns'] * data['Signal'].shift(1)
    data.dropna(inplace=True)
    
    # We show the results:
    return mean_squared_error(data['Close'], data['MAFast']),mean_squared_error(data['Close'], data['MASlow']),r2_score(data['Close'], data['MAFast']),r2_score(data['Close'], data['MASlow'])

def linear_reg(source):
    data = pd.read_csv(source)
    data = data.tail(1000)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    data['Day'] = data.index.day
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    
    X = data[['Day', 'Month', 'Year']].values
    y = data['Close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.legend()
    plt.title('Stock Price Prediction with Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show
    
    mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse,r2

import numpy as np


def svm(source):
    data = pd.read_csv(source)
    # data = data.tail(1000)
    # data1 = data.reset_index()['Close']
    # data2 = data.reset_index()['Low']
    # data3 = data.reset_index()['High']
    # data4 = data.reset_index()['Open']
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    data['Day'] = data.index.day
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    
    data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    
    X = data[['Day', 'Month', 'Year']].values
    y = data['Price_Up'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy,f1

accuracy,f1=svm('MMM.csv')
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

def metrics_plot(dataset,mse_lstm,r2_lstm):
    mse_gdm,r2_gdm= gbm(dataset)
    mse_arima,r2_arima = arima(dataset)
    mse_randomforest,r2_randomforest = randomForest(dataset)
    mse_mlp,r2_mlp = MLP(dataset)
    mse_mahigh,mse_malow,r2_mahigh,r2_malow = MA_Strategy(dataset,42,252)
    mse_lr,r2_lr = linear_reg(dataset)
    mse_svm,r2_svm = svm(dataset)

    dct_mse = {"gdm":mse_gdm,"arima":mse_arima,"random forest":mse_randomforest,"Multilayer perception":mse_mlp,"LSTM":mse_lstm,"Moving Average High":mse_mahigh,"Moving Average Low":mse_malow,"Linear Regression":mse_lr,"svm(Accuracy)":mse_svm}
    dct_r2 = {"gdm":r2_gdm,"arima":r2_arima,"random forest":r2_randomforest,"Multilayer perception":r2_mlp,"LSTM":r2_lstm,"Moving Average High":r2_mahigh,"Moving Average Low":r2_malow,"Linear Regression":r2_lr,"SVM(f1 score)":r2_svm}
    x1 = list(dct_mse.keys())
    # print(x1)
    y1 = list(dct_mse.values())
    # print(y1)
    y2 = list(dct_r2.values())
    # print(y2)
    # plt.bar(x1,y1)
    # plt.xlabel('Machine learning Model')
    # plt.ylabel("Mean Squared Error")
    # plt.savefig("metric1_plot0.png")
    # st.image("metric1_plot0.png")
    # plt.bar(x1,y2)
    # plt.xlabel('Machine learning Model')
    # plt.ylabel("R2 Score")
    # plt.savefig("metric1_plot1.png")

    # st.image("metric1_plot1.png")
    st.write(pd.DataFrame({
        "Algorithm":x1,
        "Mean Squared Error":y1
    }))
    st.write(pd.DataFrame({
        "Algorithm":x1,
        "R2 Score":y2
    }))
    # st.write(dct_r2)
    


selected = st.selectbox(label="Select Company",options=names,)
cur = 'MSFT'
if(cur==selected):
    fetch_real_time_data(cur)
    st.title("Prediction for {}".format(selected))
    lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4,mse_lstm,r2_lstm = lstm(dct[selected])
    make_plot(dct[selected],lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4)
    print(mse_lstm,r2_lstm)
    metrics_plot("{}.csv".format(cur),mse_lstm,r2_lstm)
else:
    cur = dct[selected]
    fetch_real_time_data(cur)
    st.title("Prediction for {}".format(selected))
    lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4,mse_lstm,r2_lstm = lstm(dct[selected])
    print(mse_lstm,r2_lstm)
    make_plot(dct[selected],lst_output_d1,lst_output_d2,lst_output_d3,lst_output_d4)
    metrics_plot("{}.csv".format(cur),mse_lstm,r2_lstm)

