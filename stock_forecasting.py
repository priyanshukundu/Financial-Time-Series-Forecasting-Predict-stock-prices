import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='5y')  # Get last 5 years of data
        df = df[['Close']].rename(columns={'Close': 'price'})
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Load data
ticker = "AAPL"
df = get_stock_data(ticker)
if df is None:
    raise SystemExit("Failed to fetch stock data")

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit SARIMAX Model
sarima_model = SARIMAX(train.price, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit()
print("SARIMA model fitted.")

# Fit GARCH Model
garch_model = arch_model(train.price.pct_change().dropna(), vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')
print("GARCH model fitted.")

# LSTM Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df.price.values.reshape(-1, 1))

X_train, y_train = [], []
for i in range(60, train_size):
    X_train.append(data_scaled[i-60:i, 0])
    y_train.append(data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# LSTM Model
lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
print("LSTM model trained.")

# Forecasting
sarima_forecast = sarima_result.forecast(steps=len(test))
garch_forecast = garch_result.forecast(start=len(train), horizon=len(test)).variance.mean(axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df.price, label='Actual Price', color='blue')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='red')
plt.legend()
plt.show()
