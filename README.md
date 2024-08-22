# Stock Analysis and Prediction

## Overview

This project performs stock analysis and prediction for a list of tech stocks using historical data. It includes data visualization, calculation of Exponential Moving Averages (EMA), and prediction using both Long Short-Term Memory (LSTM) and feedforward neural networks.

## Setup

### Prerequisites

- Python 3.x
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `pandas_datareader`
  - `yfinance`
  - `keras`
  - `scikit-learn`

### Installation

To install the necessary libraries, use the following pip commands:

```bash
pip install pandas numpy matplotlib seaborn pandas_datareader yfinance keras scikit-learn
```
# Usage
Importing Libraries
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
```

# Data Download and Preparation
- **Specify the List of Tech Stocks**
```bash
tech_list = ['PNB.NS', 'IRCTC.NS', 'IRFC.NS']```

-** Set up Start and End Dates**
```bash
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
```

- **Download Data**
```bash
company_list = []
for stock in tech_list:
    company_data = pdr.get_data_yahoo(stock, start, end)
    company_data['Company'] = stock
    company_list.append(company_data)
df = pd.concat(company_list)
df.reset_index(inplace=True)
```

# Data Visualisation

- **Plot Closing Prices**
```bash
plt.figure(figsize=(15, 10))
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.title(f"Closing Price of {tech_list[i - 1]}")
plt.tight_layout()
```

-**PLot Sales Volume**
```bash
plt.figure(figsize=(15, 10))
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.title(f"Sales Volume for {tech_list[i - 1]}")
plt.tight_layout()
```

# Exponential Moving Average (EMA)

-**Calculate and Plot EMA**
```bash
ema_day = [10, 20, 50]

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

for ema in ema_day:
    for company in company_list:
        column_name = f"EMA for {ema} days"
        company[column_name] = calculate_ema(company['Adj Close'], ema)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
for i, (company, symbol) in enumerate(zip(company_list, tech_list)):
    company[['Adj Close', f'EMA for 10 days', f'EMA for 20 days', f'EMA for 50 days']].plot(ax=axes[i])
    axes[i].set_title(symbol)
plt.tight_layout()
```

# Predictive Modelling

-**Prepare Data for LSTM**

```bash
data = symbol_data.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset)*.95))
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
```

-**Train LSTM Model**
```bash
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)
```

-**Evaluate Model**
```bash
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
```

-**Train FeedForward Neural Network Model**
```bash
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)
```

# ACKNOWLEDGEMENTS
- **The data is sourced from Yahoo Finance.**
-**Libraries used:** pandas, numpy, matplotlib, seaborn, pandas_datareader, yfinance, keras, scikit-learn.
sql

