import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

df_A = pd.read_csv('samsung.csv')
training_set = df_A.iloc[:1556, 4:5].values

#print(training_set)



# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialising the LSTM
clf = Sequential()


clf.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
clf.add(Dropout(0.2))


clf.add(LSTM(units = 50, return_sequences = True))
clf.add(Dropout(0.2))


clf.add(LSTM(units = 50, return_sequences = True))
clf.add(Dropout(0.2))


clf.add(LSTM(units = 50))
clf.add(Dropout(0.2))

clf.add(Dense(units = 1))


clf.compile(optimizer = 'adam', loss = 'mean_squared_error')


clf.fit(X_train, y_train, epochs = 100, batch_size = 32)

with open('LSTM.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('LSTM.pickle','rb')
clf = pickle.load(pickle_in)

df_A_test = df_A.iloc[1556:, 4:5].values
real_stock_price = df_A_test

dataset_total = df_A['Close']
inputs = dataset_total[len(dataset_total) - len(df_A_test) - 60:].values
#print(inputs)

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = clf.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Samsung Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Samsung Stock Price')
plt.title('Samsung Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Samsung Stock Price')
plt.legend()
plt.show()
plt.rc('axes', grid=True)
plt.rc('figure', figsize=(12, 8))


#linear reg

df = pd.read_csv('samsung.csv')

df_A_test = df.iloc[1556:, 4:5].values
real_stock_price = df_A_test

#print(df.tail())

df = df[['Close']]
forecast_out = int(664) # predicting 664 days into future
df['Prediction'] = df[['Close']].shift(-forecast_out)
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)



X_forecast = X[-forecast_out:]
X = X[:-forecast_out]


y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.425)


model = LinearRegression()
model.fit(X_train,y_train)


forecast_prediction = model.predict(X_forecast)



plt.plot(real_stock_price, color = 'red', label = 'Real Samsung Stock Price')
plt.plot(forecast_prediction, color = 'green', label = 'Predicted Samsung Stock Price by Linear Regression')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Samsung Stock Price by LSTM')
plt.title('Samsung Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Samsung Stock Price')
plt.legend()
plt.show()
plt.rc('axes', grid=True)
plt.rc('figure', figsize=(12, 8))
