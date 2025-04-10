Stock Price Forecasting with SARIMA, GARCH, and LSTM
Overview

This project implements a stock price forecasting model using three different time series techniques:
SARIMA (Seasonal AutoRegressive Integrated Moving Average): For capturing trend and seasonality.
GARCH (Generalized AutoRegressive Conditional Heteroskedasticity): For modeling volatility.
LSTM (Long Short-Term Memory Neural Network): For learning complex patterns in stock price data.

Features
Fetches 5 years of historical stock data using yfinance.
Implements SARIMA for trend and seasonality forecasting.
Uses GARCH for volatility modeling.
Applies LSTM for deep learning-based time series forecasting.
Visualizes actual stock prices along with SARIMA forecasts.

Requirements
Before running the script, install the required dependencies:
pip install yfinance pandas numpy matplotlib statsmodels arch scikit-learn tensorflow

How It Works
Fetch Stock Data: Retrieves 5 years of closing price data from Yahoo Finance.
Train-Test Split: Uses 80% of the data for training and the rest for testing.
SARIMA Model: Trains a seasonal ARIMA model and generates future forecasts.
GARCH Model: Fits a GARCH model to capture stock volatility.
LSTM Model: Preprocesses data using MinMaxScaler and trains an LSTM network.
Forecasting: Uses SARIMA and GARCH to forecast stock movements.
Visualization: Plots actual vs. predicted stock prices.

How to Run
Run the script using:
python stock_forecasting.py
The default stock ticker is AAPL (Apple Inc.). You can change it by modifying the ticker variable in the script.

Results
The SARIMA model provides trend-based price forecasts.
The GARCH model predicts future volatility.
The LSTM model learns stock price patterns and makes sequence-based predictions.

Limitations
LSTM requires significant computational power for larger datasets.
SARIMA assumes stationarity and predefined seasonal patterns.
GARCH focuses only on volatility, not price direction.

Future Improvements

Implement hybrid models combining SARIMA and LSTM.
Add real-time stock price predictions.
Fine-tune LSTM hyperparameters for better accuracy.

License
This project is open-source and available for modification and improvement.

Developed by Priyanshu Kundu : )

