#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
from datetime import date
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from finta import TA


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
    rsi = TA.RSI(data['Adj Close'])
    macd = TA.MACD(data['Adj Close'])
    macd_values = macd.macd
    signal_values = macd.signal

    # Initialize variables to keep track of trades
    buys = []
    sells = []

    # Iterate through the data and execute trades
    for i in range(1, len(data)):
        # RSI strategy
        if strategy == 'RSI':
            if rsi[i] < 30:
                buys.append(i)
            elif rsi[i] > 70:
                sells.append(i)
        
        # MACD strategy
        elif strategy == 'MACD':
            if macd.macd[i] < macd.signal[i] and macd.macd[i-1] > macd.signal[i-1]:
                buys.append(i)
            elif macd.macd[i] > macd.signal[i] and macd.macd[i-1] < macd.signal[i-1]:
                sells.append(i)
                

    # Plot the trades
    plt.plot(data['Close'], '-', label='Close Price')
    plt.plot(rsi, '-', label='RSI')
    plt.plot(macd.macd, '-', label='MACD')
    plt.scatter(buys, data.loc[buys, 'Close'], color='green', marker='^',label='Buy')
    plt.scatter(sells, data.loc[sells, 'Close'], color='red', marker='v',label='Sell')
    plt.legend()
    plt.show()
    
execute_trade(data,selected_strategy)


    


# In[ ]:




