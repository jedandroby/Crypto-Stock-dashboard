import streamlit as st
from datetime import date
import datetime
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib
# matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('docs')
from functions import get_data_yahoo
from functions import analyze_data
from functions import monte_carlo_sim
from functions import LSTM_model
from functions import trading_algo
from functions import linear_model
from functions import prop_model
from finta import TA
import ta
from talib import abstract

# Use the full page instead of a narrow central column
# st.set_page_config(layout="wide")

def monsim():
    st.title("Monte Carlo Asset Predictor")
    # get data with yahoo api function from function.py
    data = get_data_yahoo()
    
    # analyze the data with function from functions.py
    analyze_data(data)
    
    # run simulation with function from functions.py 
    monte_carlo_sim(data)
    
    st.write("Disclaimer")
    st.caption("""The values that are displayed in this dashboard are solely there for the purpose of knowledge and education. This in no
        way is financial advice, and we strongly recommend to take into account many other factors before entering a trade. With that being said,
        we hope you found this information helpful, and we wish you the best of luck on your trading endeavours!""")

def intro ():
    image = Image.open('logo.jpg')
    st.image(image, width = 750)
    st.title("Crypto App")
    st.write('Welcome to our advanced financial platform, designed to provide you with the tools and insights you need to make informed investment decisions. Our platform combines cutting-edge predictive models, such as Monte Carlo simulations, machine learning, and algorithmic trading, with a wealth of historical market data, to provide unparalleled insights into the performance of a wide range of assets.'
    "Our advanced models include a Monte Carlo asset predictor, a time series predictor, a backtesting feature for your trading indicators, and a logistic regression model. These tools allow you to test and optimize your investment strategies, as well as gain a deeper understanding of the underlying factors that affect asset prices."
    "Whether you're a professional trader, a seasoned investor, or just starting out, our platform can help you make better-informed decisions. By providing you with the latest predictive tools and a wealth of historical data, our platform can give you the edge you need to succeed in today's fast-paced financial markets."
    "Experience the difference that advanced predictive tools can make in your financial success. Sign up for our platform today and gain access to the insights and tools you need to make informed investment decisions.")
    st.title("Pick a coin for Analisys")

    stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
    dropdown = st.multiselect('Pick your coin',stocks)

    start = st.date_input('Start',value=pd.to_datetime("2010-01-01"))
    end = st.date_input('End', value=pd.to_datetime('today'))

    def relativeret(df):
        rel = df.pct_change()
        cumret = (1+rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret
    

    if len(dropdown) > 0:
        df = relativeret(yf.download(dropdown,start,end)['Close'])
        st.header('Return of {}' .format(dropdown))
        st.line_chart(df)

    
def ML ():
    st.title("LSTM (Long Short-Term Memory) Predictor")
    data = get_data_yahoo()

    # ask the the user to select variables for model - lookback, neruons, epochs
    st.sidebar.title("Look Back Window")
    look_back = int(st.sidebar.slider("How many days do you want to predict:",1,150))
    st.sidebar.title("Amount of Neurons")
    neurons = int(st.sidebar.slider("How many neurons do you want", 6,64))
    st.sidebar.title('Amount of Epochs')
    epoch = int(st.sidebar.slider('How many Epochs do you want to run through?(The higher the number, the longer it will take)',1,10))
    st.sidebar.title('Train/test split size')
    test_split = float(st.sidebar.slider('What percent do you want the model to train and validate on?', 0.75,0.95))
    
    if st.sidebar.button('Predict'):
        with st.spinner('Wait for it...'):  
            st.balloons()
            LSTM_model(look_back, neurons, epoch, data, test_split)
    
    st.write("Disclaimer")
    st.caption("""The values that are displayed in this dashboard are solely there for the purpose of knowledge and education. This in no
        way is financial advice, and we strongly recommend to take into account many other factors before entering a trade. With that being said,
        we hope you found this information helpful, and we wish you the best of luck on your trading endeavours!""")
    

def prop():
# Forecasting
    st.title("Prophet Asset Predictor")
    data = get_data_yahoo()
    
    st.sidebar.title('How many days do you want to predict?')
    n_days = int(st.sidebar.slider("Days of Prediction:",1,7))
    period = n_days * 365
    
    prop_model(data,period)
    st.write("Disclaimer")
    st.caption("""The values that are displayed in this dashboard are solely there for the purpose of knowledge and education. This in no
        way is financial advice, and we strongly recommend to take into account many other factors before entering a trade. With that being said,
        we hope you found this information helpful, and we wish you the best of luck on your trading endeavours!""")
def back_testing():
    
    data = get_data_yahoo()

    trading_algo(data)
    
def linear_reg():
    df = get_data_yahoo()

    st.title('Explaining the model')
    st.write('Linear regression is a statistical method that is used to model the relationship between a dependent variable and one or more independent variables. The goal of linear regression is to find the best linear relationship that explains the relationship between the dependent variable and the independent variable.')
    st.title('Explaining the strategy')
    st.write('The first step is creating a column where we can calculate the percent change on the Closing price'
    'so that the returns can be normally distributed. We then create a function that creates columns (lag1,lag2...) that it shifts the returns in each column'
    'We then run the LinearRegression model where we fit the lags and the return'
    'Finally, we add a new column called, direction, which we assign value of 1 to the rows where the prediction is greater than 0 and -1 to the rows where the prediction is less than or equal to 0. It then creates a new column called (strategy) which multiplies the (direction) column with the (returns) column.' 
    'This creates a simple trading strategy where a positive prediction results in a buy signal and a negative prediction results in a sell signal.' )

    linear_model(df)
    st.write("Disclaimer")
    st.caption("""The values that are displayed in this dashboard are solely there for the purpose of knowledge and education. This in no
        way is financial advice, and we strongly recommend to take into account many other factors before entering a trade. With that being said,
        we hope you found this information helpful, and we wish you the best of luck on your trading endeavours!""")


page_names_to_funcs = {
    'Home Page': intro,
    "Monte Carlo Simulator": monsim,
    'Long Short-Term Memory Model':ML,
    'Prophet':prop,
    'Trading Backtester':back_testing,
    'Linear Regression Model':linear_reg
    
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()