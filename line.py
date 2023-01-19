import streamlit as st
from datetime import date
import datetime
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn import svm


st.title("Linear Regression Asset Predictor")

stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
selected_stocks = st.selectbox("Pick a coin for prediction",stocks)

@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker,"2010-01-01")
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stocks)
st.subheader('Data Head')
st.write(data.head())
st.subheader('Data Tail')
st.write(data.tail())
    
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

df = data
df['returns'] = np.log(df.Close.pct_change()+1)
def lagit (df,lags):
    names = []
    for i in range(1,lags +1):
        df['Lag_'+ str(i)] = df['returns'].shift(i)
        names.append('Lag_'+ str(i))
    return names

lagnames = lagit(df,5)
df.dropna(inplace=True)

model = LinearRegression()
model.fit(df[lagnames],df['returns'])
df['prediction'] = model.predict(df[lagnames])

df['direction'] = [1 if i > 0 else -1 for i in df.prediction]
df['strategy'] = df['direction'] * df['returns']

x = df['returns'].cumsum()
y = df['strategy'].cumsum()

st.title('Explaining the model')
st.write('Linear regression is a statistical method that is used to model the relationship between a dependent variable and one or more independent variables. The goal of linear regression is to find the best linear relationship that explains the relationship between the dependent variable and the independent variable.')
st.title('Explaining the strategy')
st.write('The first step is creating a column where we can calculate the percent change on the Closing price'
'so that the returns can be normally distributed. We then create a function that creates columns (lag1,lag2...) that it shifts the returns in each column'
'We then run the LinearRegression model where we fit the lags and the return'
'Finally, we add a new column called, direction, which we assign value of 1 to the rows where the prediction is greater than 0 and -1 to the rows where the prediction is less than or equal to 0. It then creates a new column called (strategy) which multiplies the (direction) column with the (returns) column.' 
'This creates a simple trading strategy where a positive prediction results in a buy signal and a negative prediction results in a sell signal.' )

st.subheader('Actual Returns vs Strategy Returns')
if st.button('Click to see Dataframe'):
    st.write(df.head(10))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=x, name='Actual Returns', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=y, name='Strategy Returns', line=dict(color='red')))
fig.update_layout(title='Comparison of Actual Returns and Strategy Returns',xaxis_title='Date',yaxis_title='Returns')
if st.button('Click here to see the plot'):
    st.plotly_chart(fig)