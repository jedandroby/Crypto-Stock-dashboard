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


trading_meth = ("RSI",'MACD')
selected_strategy = st.selectbox("Pick a Trading Method",trading_meth)

strategy = (selected_strategy)
def plotting(strategy):
    if strategy == "RSI":

        def execute_trade(data, strategy):
            # Initialize indicators
            rsi = TA.RSI(data)
    

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

            
        execute_trade(data,selected_strategy)
    else:
        def execute_trades(data, strategy):
    # Initialize indicators
           
            macd, macd_signal, macd_hist = abstract.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            macd_df = pd.DataFrame(macd, columns=['MACD'])

            # Initialize variables to keep track of trades
            buy_indices = []
            buy_closes = []
            sell_indices = []
            sell_closes = []

            # Iterate through the data and execute trades
            for i in range(1, len(data)):
                if strategy == 'MACD':
                    if macd[i] < macd_signal[i] and macd[i-1] > macd_signal[i-1]:
                        buy_indices.append(i)
                        buy_closes.append(data.loc[i, 'Close'])
                    elif macd[i] > macd_signal[i] and macd[i-1] < macd_signal[i-1]:
                        sell_indices.append(i)
                        sell_closes.append(data.loc[i, 'Close'])

            # Plot MACD buys and sells
            plt.plot(data['Close'], '-', label='Close Price')
            plt.scatter(buy_indices, buy_closes, color='green', marker='^', label='Buy')
            plt.scatter(sell_indices, sell_closes, color='red', marker='v', label='Sell')
            plt.legend()
            plt.title('MACD Strategy')
            st.pyplot()
    
        execute_trades(data,selected_strategy)
plotting(strategy)





# Using object notation
initial_capital = st.sidebar.selectbox(
    "How much money would you like to invest?",
    ("10",'20',"100",'200','300','400')
)
share_size = st.sidebar.selectbox(
    "How many shares would you like to purchase on each buy signal?",
    ("10",'20',"100",'200','300','400')
)
def stats(strategy):
        
    if strategy == "RSI":
            
        def backtest_RSI(data, initial_capital, share_size):
            initial_capital = int(initial_capital)
            share_size = int(share_size)
            # calculate RSI
            
            rsi = TA.RSI(data)

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
                        buy_indices.append(data.loc[i, 'Date'])
                        buy_closes.append(data.loc[i, 'Close'])
                
                    elif rsi[i] > 70:
                        sell_indices.append(data.loc[i, 'Date'])
                        sell_closes.append(data.loc[i, 'Close'])
            # create signals dataframe
            signals_df = data.loc[:,["Close"]]
            signals_df['Signal'] = 0.0
            signals_df['Trade Type'] = np.nan
            signals_df['Total Cost'] = 0.0
            signals_df['Total Revenue'] = 0.0
            signals_df['Net Profit/Loss'] = 0.0
            # initialize previous price
            previous_price = 0
            # initialize share size and accumulated shares
            accumulated_shares = 0
            invested_capital = 0
            roi = 0
            buy_indices = pd.Series(buy_indices)
            buy_closes = pd.Series(buy_closes)
            sell_indices = pd.Series(sell_indices)
            sell_closes = pd.Series(sell_closes)
           
            buy_df = pd.DataFrame(buy_closes)
            buy_df.index = buy_indices
            
            sell_df=pd.DataFrame(sell_closes)
            sell_df.index= sell_indices
            
            df1 = pd.concat([buy_df, sell_df], axis=1, keys=['buy', 'sell'])
            # df1['group'] = df1.index.to_series().groupby(df1.index.to_series().diff().ne(0).cumsum()).cumcount()
            # df1 = df1.pivot(columns='group', values=['buy', 'sell'])
            # df1.columns = ['buy', 'sell']
            st.write(df1)
            df2 = pd.DataFrame()
            prev_signal = None
            flag = None
            # iterate through the rows of the dataframe
            for i, row in df1.iterrows():
                if flag == 'buy':
                    if row['sell'].notna().any():
                        df2 = df2.append({'Date': i, 'Signal': 'Sell', 'Price': row['sell'].values}, ignore_index=True)
                        flag = 'sell'
                else:
                    if row['buy'].notna().any():
                        df2 = df2.append({'Date': i, 'Signal': 'Buy', 'Price': row['buy'].values}, ignore_index=True)
                        flag = 'buy'
            
            def convert_to_float(x):
                return round(float(x), 2)
            df2['Price'] = df2['Price'].apply(convert_to_float)
            st.write(df2)



                 
            # buy_df.set_index("Date", inplace=True)
            # sell_df = pd.concat([sell_indices, sell_closes], axis=1)
            # sell_df.set_index("Date", inplace=True)
            df2['Total Cost'] = 0.0
            df2['Total Revenue'] = 0.0
            initial_capital = int(initial_capital)
            share_size = int(share_size)
            
            for i in range(0, len(df2)):
                # buy if RSI < 30
                if df2.loc[i, 'Signal'] == 'Buy': 
                    # Buy condition
                    # signals_df.loc[i, 'Signal'] = 1.0
                    # signals_df.loc[i, 'Trade Type'] = "Buy"
                    # calculate the cost of the trade
                    cost = -(df2.loc[i, 'Price'] * share_size)
                    df2.loc[i, 'Total Cost'] += cost
                    # add the number of shares purchased to the accumulated shares
                    # accumulated_shares += share_size
                
                # sell if RSI > 70
                elif df2.loc[i, 'Signal'] == 'Sell': 
                    # Sell condition
                    # signals_df.loc[i, 'Signal'] = -1.0
                    # signals_df.loc[i, 'Trade Type'] = "Sell"
                    # calculate the proceeds of the trade
                    revenue = (df2.loc[i, 'Price'] * share_size)
                    df2.loc[i, 'Total Revenue'] += revenue
                    # reset accumulated shares
                    # accumulated_shares = 0
                    
                # no signal if RSI between 30 and 70
                else:
                    # signals_df.loc[i, 'Signal'] = signals_df.loc[i-1, 'Signal']
                    # signals_df.loc[i, 'Trade Type'] = signals_df.loc[i-1, 'Trade Type']
                    # signals_df.loc[i, 'Net Profit/Loss'] = signals_df.loc[i-1, 'Net Profit/Loss']
                    skip

            # calculate net profit/loss
            # signals_df['Net Profit/Loss'] = signals_df['Total Revenue'] - signals_df['Total Cost']
            total_revenue = df2['Total Revenue'].sum()
            total_cost = df2['Total Cost'].sum()
            st.write(df2)




        # calculate net profit/loss
            
            # signals_df['Net Profit/Loss'] = signals_df['Portfolio Cash'] - initial_capital
            # calculate total profit/loss
            # total_profit_loss = round(signals_df["Net Profit/Loss"].iloc[-1], 2)
            # st.write(f"The total profit/loss of the trading strategy is ${total_profit_loss}.")
                    # Print the ROI
            # invested_capital = invested_capital + abs(signals_df["Cost/Proceeds"].sum())
            roi = round(total_revenue / -(total_cost) * 100, 2)
            st.write(f"The trading algorithm resulted in a return on investment of {roi}%")
            st.write(f"The total money invested was ${total_cost}")
            st.write(f"The total Revenue generated was ${total_revenue}")
        backtest_RSI(data, initial_capital, share_size)



    else:
                
        def backtest_MACD(data, initial_capital, share_size):
            # Initialize variables to keep track of trades
            buy_indices = []
            buy_closes = []
            sell_indices = []
            sell_closes = []
            macd, macd_signal, macd_hist = abstract.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            macd_df = pd.DataFrame(macd, columns=['MACD'])
            data = data
            # Iterate through the data and execute trades
            for i in range(1, len(data)):
                if strategy == 'MACD':
                    if macd[i] < macd_signal[i]:
                        buy_indices.append(data.loc[i, 'Date'])
                        buy_closes.append(data.loc[i, 'Close'])
                    elif macd[i] > macd_signal[i]:
                        sell_indices.append(data.loc[i, 'Date'])
                        sell_closes.append(data.loc[i, 'Close'])
            buy_indices = pd.Series(buy_indices)
            buy_closes = pd.Series(buy_closes)
            sell_indices = pd.Series(sell_indices)
            sell_closes = pd.Series(sell_closes)
           
            buy_df = pd.concat([sell_closes, buy_closes], axis=1)
            buy_df = buy_df.rename(columns={0: "Sell Closes", 1: "Buy Closes"})
            # buy_df.set_index("Date", inplace=True)
            # sell_df = pd.concat([sell_indices, sell_closes], axis=1)
            # sell_df.set_index("Date", inplace=True)
            initial_capital = int(initial_capital)
            share_size = int(share_size)
            # calculate RSI
            
            # create signals dataframe
            signals_df = data.loc[:,["Close"]]
            signals_df['Signal'] = 0.0
            signals_df['Trade Type'] = np.nan
            signals_df['Total Cost'] = 0.0
            signals_df['Total Revenue'] = 0
            signals_df['Net Profit/Loss'] = 0.0
            # initialize previous price
            previous_price = 0
            accumulated_shares = 0
            invested_capital = 0
            # initialize share size and accumulated shares
        

            
            for i in range(1, len(buy_df)): 
                        if buy_df.loc[i, 'Buy Closes'] > 1:
                            # Buy condition
                            signals_df.loc[i, 'Signal'] = 1.0
                            signals_df.loc[i, 'Trade Type'] = "Buy"
                            # calculate the cost of the trade
                            cost = -(data.loc[i, 'Close'] * share_size)
                            signals_df.loc[i, 'Total Cost'] += cost
                            # update portfolio cash and holdings
                            # signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash'] + cost
                            # signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Portfolio Cash']
                            # add the number of shares purchased to the accumulated shares
                            accumulated_shares += share_size
                            
                        elif buy_df.loc[i, 'Sell Closes'] > 1:
                            # Sell condition

                            signals_df.loc[i, 'Signal'] = -1.0
                            signals_df.loc[i, 'Trade Type'] = "Sell"
                            # calculate the proceeds of the trade
                            revenue = (data.loc[i, 'Close'] * share_size)
                            signals_df.loc[i, 'Total Revenue'] += revenue
                            # update portfolio cash and holdings
                            # signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash'] + proceeds
                            # signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i, 'Portfolio Cash']
                            # reset accumulated shares
                            accumulated_shares = 0
                            # no signal if RSI between 30 and 70
                            
                        else:
                            signals_df.loc[i, 'Signal'] = 0
                            signals_df.loc[i, 'Trade Type'] = 'Hold'
                            # signals_df.loc[i, 'Cost/Proceeds'] = signals_df.loc[i-1, 'Cost/Proceeds']
                            # signals_df.loc[i, 'Portfolio Holdings'] = signals_df.loc[i-1, 'Portfolio Holdings']
                            # signals_df.loc[i, 'Portfolio Cash'] = signals_df.loc[i-1, 'Portfolio Cash']
                            signals_df.loc[i, 'Net Profit/Loss'] = signals_df.loc[i-1, 'Net Profit/Loss']
            
                    # Print the results
            total_revenue = signals_df['Total Revenue'].sum()
            total_cost = signals_df['Total Cost'].sum()
                    # calculate net profit/loss
            # signals_df['Net Profit/Loss'] = signals_df['Portfolio Cash'] - initial_capital
            #         # calculate total profit/loss
            # total_profit_loss = round(signals_df["Net Profit/Loss"].iloc[-1], 2)
            # st.write(f"The total profit/loss of the trading strategy is ${total_profit_loss}.")
                    # Print the ROI
    
            # invested_capital = initial_capital + abs(signals_df["Cost/Proceeds"].sum())
            roi = round(total_revenue / -(total_cost) * 100, 2)
            st.write(f"The trading algorithm resulted in a return on investment of {roi}%")

            st.write(signals_df)
        backtest_MACD(data, initial_capital, share_size)
            

stats(strategy)