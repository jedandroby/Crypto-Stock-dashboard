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

# initial_capital = st.number_input("Enter your initial trading capital: ")
# share_size = st.number_input("Enter your initial share size: ")

# def backtest(data, strategy, initial_capital, share_size):
#     rsi = TA.RSI(data)
#     signals_df = data.loc[:,["Close"]]
#     signals_df['Signal'] = 0.0
#     signals_df['Buys'] = np.nan
#     signals_df['Sells'] = np.nan
#     signals_df['Position'] = 0.0
#     signals_df['Portfolio Holdings'] = initial_capital
#     signals_df['Net Profit/Loss'] = 0.0
#     signals_df['Entry/Exit'] = np.nan
#     signals_df['Portfolio Cash'] = initial_capital
#     signals_df['Portfolio Total'] = initial_capital
#     signals_df['Portfolio Daily Returns'] = 0.0
#     signals_df['Portfolio Cumulative Returns'] = 1.0

#     if strategy == 'RSI':
#         for i in range(1, len(data)):
#             if rsi[i] < 30 and rsi[i-1] >= 30: 
#                 # Buy condition
#                 signals_df.loc[i, 'Signal'] = 1.0
#                 signals_df.loc[i, 'Buys'] = data.loc[i, 'Close']
#                 signals_df.loc[i, 'Position'] += share_size
#                 signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Position'] * data.loc[i, 'Close']
#                 signals_df.loc[i, 'Portfolio Cash'] = initial_capital - (signals_df.loc[i, 'Portfolio Holdings'])
#             elif rsi[i] > 70 and rsi[i-1] <= 70: 
#                 # Sell condition
#                 signals_df.loc[i, 'Signal'] = -1.0
#                 signals_df.loc[i, 'Sells'] = data.loc[i, 'Close']
#                 signals_df.loc[i, 'Position'] -= share_size
#                 signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Position'] * data.loc[i, 'Close']
#                 signals_df.loc[i, 'Portfolio Cash'] = initial_capital - (signals_df.loc[i, 'Portfolio Holdings'])
#                 signals_df.loc[i, 'Net Profit/Loss'] = (data.loc[i, 'Close'] - data.loc[i-1, 'Close']) * share_size
#             else:
#             # signals_df.loc[i, 'Signal'] = signals_df.loc[i-1, 'Signal']
#                 signals_df.loc[i, 'Position'] = signals_df.loc[i-1, 'Position']
#                 signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i-1, 'Portfolio Holdings']
#                 signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash']
#                 signals_df.loc[i, 'Net Profit/Loss'] = signals_df.loc[i-1, 'Net Profit/Loss']
#         signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']
#         signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()
#         signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1
#                 # st.write(signals_df)

#     st.write(signals_df)

# backtest(data, strategy, initial_capital, share_size)


# Using object notation
initial_capital = st.sidebar.selectbox(
    "How much money would you like to invest?",
    ("10",'20',"100",'200','300','400')
)
share_size = st.sidebar.selectbox(
    "How many shares would you like to purchase on each buy signal?",
    ("10",'20',"100",'200','300','400')
)

def backtest_RSI(data, initial_capital, share_size):
    initial_capital = int(initial_capital)
    share_size = int(share_size)
    # calculate RSI
    rsi = TA.RSI(data)
    data = data
    # create signals dataframe
    signals_df = data.loc[:,["Close"]]
    signals_df['Signal'] = 0.0
    signals_df['Trade Type'] = np.nan
    signals_df['Cost/Proceeds'] = np.nan
    signals_df['Portfolio Holdings'] = initial_capital
    signals_df['Portfolio Cash'] = initial_capital
    signals_df['Net Profit/Loss'] = 0.0
    # initialize previous price
    previous_price = 0
    # initialize share size and accumulated shares
    accumulated_shares = 0
    invested_capital = 0
    roi = 0
    for i in range(1, len(data)):
    # buy if RSI < 30
        if rsi[i] < 30: 
            # Buy condition
            signals_df.loc[i, 'Signal'] = 1.0
            signals_df.loc[i, 'Trade Type'] = "Buy"
            # calculate the cost of the trade
            cost = -(data.loc[i, 'Close'] * share_size)
            signals_df.loc[i, 'Cost/Proceeds'] = cost
            # update portfolio cash and holdings
            signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash'] + cost
            signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Portfolio Cash']
            # add the number of shares purchased to the accumulated shares
            accumulated_shares += share_size
        
        # sell if RSI > 70
        elif rsi[i] > 70: 
            # Sell condition
            signals_df.loc[i, 'Signal'] = -1.0
            signals_df.loc[i, 'Trade Type'] = "Sell"
            # calculate the proceeds of the trade
            proceeds = (data.loc[i, 'Close'] * accumulated_shares)
            signals_df.loc[i, 'Cost/Proceeds'] = proceeds
            # update portfolio cash and holdings
            signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash'] + proceeds
            signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Portfolio Cash']
            # reset accumulated shares
            accumulated_shares = 0
            
            # no signal if RSI between 30 and 70
        else:
            signals_df.loc[i, 'Signal'] = signals_df.loc[i-1, 'Signal']
            signals_df.loc[i, 'Trade Type'] = signals_df.loc[i-1, 'Trade Type']
            signals_df.loc[i, 'Cost/Proceeds'] = signals_df.loc[i-1, 'Cost/Proceeds']
            signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i-1, 'Portfolio Holdings']
            signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash']
            signals_df.loc[i, 'Net Profit/Loss'] = signals_df.loc[i-1, 'Net Profit/Loss']

                
                # Calculate the return on investment (ROI)
                

                
                
    
# calculate net profit/loss
    signals_df['Net Profit/Loss'] = signals_df['Portfolio Cash'] - initial_capital
    # calculate total profit/loss
    total_profit_loss = round(signals_df["Net Profit/Loss"].iloc[-1], 2)
    st.write(f"The total profit/loss of the trading strategy is ${total_profit_loss}.")
              # Print the ROI
    invested_capital = invested_capital + abs(signals_df["Cost/Proceeds"].sum())
    roi = round((total_profit_loss / -(invested_capital)) * 100, 2)
    st.write(f"The trading algorithm resulted in a return on investment of {roi}%")
backtest_RSI(data, initial_capital, share_size)



