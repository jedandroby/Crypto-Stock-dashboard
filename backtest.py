#!/usr/bin/env python
# coding: utf-8

# In[7]:

import matplotlib
matplotlib.use("TkAgg") 
import streamlit as st
from datetime import date
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from finta import TA
import ta
from talib import abstract

# In[15]:





# HEAD
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


def monsim():
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Monte Carlo Asset Predictor")
    # 3a92c204243b3da87ecc97f041ff03a673f631fe

    stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
    selected_stocks = st.selectbox("Pick a coin for prediction",stocks)

    # n_years = st.slider("Years of Prediction:",1,15)
    # period = n_years * 365


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker,START,TODAY)
        data.reset_index('Date',inplace=True)
        data = pd.DataFrame(data)
        return data

# In[2]:


stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
selected_stocks = st.selectbox("Pick a coin for prediction",stocks)


# In[3]:


@st.cache
def load_data(ticker):
     data = yf.download(ticker,START,TODAY)
     data.reset_index('Date',inplace=True)
     data = pd.DataFrame(data)
     return data

    # data_load_state = st.text("Load data")
data = load_data(selected_stocks)


# In[4]:
st.subheader("Asset Data Head")
st.write(data.head())
st.subheader('Asset Data Tail')
st.write(data.tail())
st.subheader("Interactive Asset Chart")
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Coin Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Coin Close'))
    fig.layout.update(xaxis_rangeslider_visible=True)     
    st.plotly_chart(fig)

plot_raw_data()
# In[5]:


trading_meth = ("RSI", 'MACD', 'VWAP',"ATR")
selected_strategy = st.selectbox("Pick a Trading Method",trading_meth)

strategy = (selected_strategy)


# In[6]:

# rsi = TA.RSI(data['Close'])
# macd = TA.MACD(data['Close'])

def execute_trade(data, strategy):
    # Initialize indicators
    rsi = TA.RSI(data)
    macd, macd_signal, macd_hist = abstract.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    macd_df = pd.DataFrame(macd, columns=['MACD'])

    # Initialize variables to keep track of trades
    buy_indices = []
    buy_closes = []
    sell_indices = []
    sell_closes = []

    # Iterate through the data and execute trades
    for i in range(1, len(data)):
        # RSI strategy
        if strategy == 'RSI':
            if rsi[i] < 30:
                buy_indices.append(i)
                buy_closes.append(data.loc[i, 'Close'])
        
            elif rsi[i] > 70:
                sell_indices.append(i)
                sell_closes.append(data.loc[i, 'Close'])
            # Plot RSI buys and sells 
    plt.plot(data['Close'], '-', label='Close Price')
    # plt.plot(rsi, '-', label='RSI')
    plt.scatter(buy_indices, buy_closes, color='green', marker='^', label='Buy')
    plt.scatter(sell_indices, sell_closes, color='red', marker='v', label='Sell')
    plt.legend()
    plt.title('RSI Strategy')
    st.pyplot()


    #     elif strategy == 'MACD':
    #         if macd[i] < macd_signal[i] and macd[i-1] > macd_signal[i-1]:
    #             buy_dates.append(data.loc[i, 'Date'])
    #             buy_closes.append(data.loc[i, 'Close'])
    #         elif macd[i] > macd_signal[i] and macd[i-1] < macd_signal[i-1]:
    #             sell_dates.append(data.loc[i, 'Date'])
    #             sell_closes.append(data.loc[i, 'Close'])

    #         # Plot MACD buys and sells
    # plt.plot(data['Close'], '-', label='Close Price')
    # plt.plot(macd_df['MACD'], '-', label='MACD')
    # plt.scatter(buy_dates, buy_closes, color='green', marker='^', label='Buy')
    # plt.scatter(sell_dates, sell_closes, color='red', marker='v', label='Sell')
    # plt.legend()
    # plt.title('MACD Strategy')
    # plt.show()

execute_trade(data,selected_strategy)


    


# In[ ]:




