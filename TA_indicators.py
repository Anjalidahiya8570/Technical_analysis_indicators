import yfinance as yf
import numpy as np
import pandas as pd

# Simple Moving Average (SMA)
def calculate_sma(data, window=20):
    return data.rolling(window=window).mean()

# Exponential Moving Average (EMA)
def calculate_ema(data, window=20):
    return data.ewm(span=window, adjust=False).mean()

# Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, signal_window)
    return macd_line, signal_line, macd_line - signal_line

# Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band

# Fetch historical price data using yfinance
def fetch_historical_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Main function to demonstrate the implementation
def main():
    ticker = 'TATAMOTORS.NS'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Fetch historical data
    data = fetch_historical_data(ticker, start_date, end_date)

    # Calculate indicators
    sma = calculate_sma(data)
    ema = calculate_ema(data)
    macd, signal, macd_histogram = calculate_macd(data)
    rsi = calculate_rsi(data)
    upper_band, lower_band = calculate_bollinger_bands(data)

    # Display results
    print("SMA:")
    print(sma.tail())

    print("\nEMA:")
    print(ema.tail())

    print("\nMACD:")
    print("MACD Line:")
    print(macd.tail())
    print("\nSignal Line:")
    print(signal.tail())
    print("\nMACD Histogram:")
    print(macd_histogram.tail())

    print("\nRSI:")
    print(rsi.tail())

    print("\nBollinger Bands:")
    print("Upper Band:")
    print(upper_band.tail())
    print("\nLower Band:")
    print(lower_band.tail())

if __name__ == "__main__":
    main()
