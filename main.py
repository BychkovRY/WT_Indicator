import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta


def RMA(source, length):
    alpha = 1 / length
    sum = np.zeros(len(source))
    sum[0] = source[0]

    for i in range(1, len(source)):
        if np.isnan(sum[i - 1]):
            sum[i] = np.mean(source[max(0, i - length + 1):i + 1])
        else:
            sum[i] = alpha*source[i] + (1 - alpha) * sum[i - 1]

    return sum

def RSI (up, down):
    rsi_calc = np.zeros(len(up))
    for i in range(0, len(up)):
        rsi_calc[i] = np.where(down[i] == 0, 100, np.where(up[i] == 0, 0, 100 - (100 / (1 + up[i] / down[i]))))
    return rsi_calc

def warrior_trend_indicator(close, long_trend_length=90, short_trend_length=33):

    array = close.to_numpy()
    change = np.diff(array)
    change = np.insert(change, 0, np.nan)
    max_change = np.maximum(change, 0)
    min_change = -np.minimum(change, 0)
    up = RMA(max_change, long_trend_length)
    down = RMA(min_change, long_trend_length)
    rsi = RSI(up, down)

    ema_rsi = ta.EMA(rsi, timeperiod=short_trend_length)
    trend = np.where(ema_rsi >= np.roll(ema_rsi, 1), "Trend is UP", "Trend is DOWN")

    return trend, ema_rsi


# Fetch stock data for AAPL from Yahoo Finance
df = yf.download("EOSE", period = "2y", interval='1d', auto_adjust=True)
df_close = df['Close']

trend, ema = warrior_trend_indicator(df_close)

df_result = pd.DataFrame({'Close': df['Close'], 'EMA': ema, 'Trend': trend})
print(df_result)

if trend[-1]==trend[-2]:
    print ('Trend Contunie')
else:
    print ('Trend Changed')
