import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential, model_from_json, load_model, save_model
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import sys

df = pd.read_csv('./data40k.csv')


# Global data pre-process
timeBase = 1113433135300  # timeBase = min(df['Global_Time'])
df['Global_Time'] = df['Global_Time'].map(lambda x: (x - timeBase) / 100)
globalXBase = 6042593  # globalXBase = min(df['Global_X'])
df['Global_X'] = df['Global_X'].map(lambda x: x - globalXBase)
globalYBase = 2133053  # globalYBase = min(df['Global_Y'])
df['Global_Y'] = df['Global_Y'].map(lambda x: x - globalYBase)


localXmax = 94  # 93.659
localYmax = 1760  # 1755.939


# Vehicle list dividing

vList = []
currentVehicleID = 0
for i in df.iterrows():
    indexVehicleID = int(i[1]['Vehicle_ID'])
    if indexVehicleID > currentVehicleID:
        currentVehicleID = indexVehicleID
        vList.append(i[0])

saveFile = open("variableSave.bin","wb")
pickle.dump(vList, saveFile)
saveFile.close()

loadFile = open("variableSave.bin", "rb")
vList = pickle.load(loadFile)
loadFile.close()

'''''
for i in range(15):
    iSlide = df[vList[i]:vList[i+1]]
    plt.plot(iSlide['Global_X'],iSlide['Global_Y'])
plt.show()
'''''

'''''''''''
df100 = df[:10]
df100xy = df100['Local_X']
columns = [df100xy.shift(i) for i in range(1, 2)]
columns.append(df100xy)
df100xy = pd.concat(columns, axis=1)
df.fillna(0, inplace=True)
print(df100xy)
'''


# timeSeries 2 Supervised
def timeseries2supervised(df, lag=1):
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# dif.
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# inv. dif.
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(np.vstack((train, test)))
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # for i in range(nb_epoch):
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, shuffle=False)
    # model.reset_states()
    return model


def forecast(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


start = vList[105]
num = 200
split = 30
df_small = df[start:start+num]
df_small_x = df_small['Local_X']


# set series
series = df_small_x

# transform data to be stationary
raw_values = series.values
dif_values = difference(raw_values, 1)
# print('\nraw_values\n', raw_values, '\ndif_values\n', dif_values)

# transform data to be supervised learning
supervised = timeseries2supervised(dif_values, 1)
spv_values = supervised.values
# print('\nsupervised_values\n', supervised.values)

# split data into train and test sets
train, test = spv_values[:-split], spv_values[-split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 100, 6)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
last_pre = 0
predictions = list()
current_data = raw_values[len(train_scaled)]
for i in range(len(test_scaled)):
    # make 1-step forecast
    if i == 0:
        X = test_scaled[i, 0:-1]
    else:
        X = np.array([last_pre])
    y = test_scaled[i, -1]
    yhat = forecast(lstm_model, 1, X)
    last_pre = yhat
    # invert scaling
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    yhat = scaler.inverse_transform(array)
    yhat = yhat[0, -1]
    # print('\nyhat after inverting scaling\n', yhat)
    # invert dif.
    yhat += current_data
    current_data = yhat
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    # print('Index=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-split:], predictions))
print('RME: %.3f' % rmse)
# line plot of GT and predicted
plt.plot(raw_values[-split:])
plt.plot(predictions)
plt.show()