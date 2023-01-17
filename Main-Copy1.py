
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('docs')
from functions import get_data_yahoo
from functions import analyze_data
from functions import monte_carlo_sim

def monsim():
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Monte Carlo Asset Predictor")
    # get data with yahoo api function from function.py
    data = get_data_yahoo(START, TODAY)
    
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
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("LSTM (Long Short-Term Memory) Predictor")
    data = get_data_yahoo(START, TODAY)
    
    
    # Forecasting
    # extract the close prices
    prices = data['Close'].values

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = scaler.fit_transform(prices.reshape(-1, 1))

    # create a time series dataset
    def create_dataset(prices, look_back=3):
        X, y = [], []
        for i in range(len(prices)-look_back-1):
            X.append(prices[i:(i+look_back), 0])
            y.append(prices[i + look_back, 0])
        return np.array(X), np.array(y)

    # use the the user to select how many days as a look-back period
    st.sidebar.title("Look Back Window")
    look_back = int(st.sidebar.slider("How many days do you want to predict:",1,150))
    # create x,y data sets
    X, y = create_dataset(prices, look_back)

    # reshape the input to be 3D [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # split the data into train and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    # get the data rdy to plot for later
    close_data = prices.reshape((-1,1))

    split_percent = 0.80
    split = int(split_percent*len(close_data))

    date_test = data['Date'][split:]

    # st.write(len(close_train))
    # st.write(close_test)
    st.sidebar.title("Amount of Neurons")
    neurons = int(st.sidebar.slider("How many neurons do you want", 4,30))
    # create the LSTM model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    st.sidebar.title('Amount of Epochs')
    epoch= int(st.sidebar.slider('How many Epochs do you want to run through?(The higher the number, the longer it will take)',1,10))
    # fit the model to the training data
    model.fit(X_train, y_train, epochs=epoch, batch_size=1, verbose=2)

    # use the model to make predictions on the test set
    y_pred = model.predict(X_test)

    # invert the predictions and true values back to the original scale
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # evaluate the model
    test_score = model.evaluate(X_test, y_test, verbose=0)

    # in streamlit app
    # st.write("Mean Squared Error: ",test_score)
    prediction = y_pred.reshape((-1))

    trace1 = go.Scatter(
        x = data['Date'],
        y = data['Close'],
        mode = 'lines',
        name = 'Actual Price'
    )
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )

    layout = go.Layout(
        title = "Real price vs Model train/test predictions ",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1,trace2], layout=layout)
    st.plotly_chart(fig)
    #predict a month of price action
    # use the model to make predictions on the data for the next month
    X_future = X[-look_back:]
    # use the trained model to make predictions on the future data
    y_future= model.predict(X_future)
    # invert the predictions back to the original scale
    y_future = scaler.inverse_transform(y_future)
    
    # Get the current date
    now = datetime.datetime.now()

    # Create a list of dates for the next 30 days
    date_list = [now + datetime.timedelta(days=x) for x in range(look_back)]

    # Convert the date list to strings in the format 'YYYY-MM-DD'
    date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
    date_strings = pd.DataFrame(date_strings, columns=['date'])
    y_future = pd.DataFrame(y_future, columns=['Predicted price'])
    y_future.index = date_strings['date']

    
    trace1 = go.Scatter(
        x = data['Date'],
        y = data['Close'],
        mode = 'lines',
        name = 'Actual Price'
    )
    trace2 = go.Scatter(
        x = date_strings['date'],
        y = y_future['Predicted price'],
        mode = 'lines',
        name = 'Predicted Data'
    )
    fig= go.Figure(data=[trace2, trace1], layout=layout)
    st.plotly_chart(fig)
    st.write(y_future)
    

def prop():
# Forecasting
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Prophet Asset Predictor")
    data = get_data_yahoo(START, TODAY)
    st.sidebar.title('How many days do you want to predict?')
    n_days = int(st.sidebar.slider("Days of Prediction:",1,7))
    period = n_days * 365
    
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds","Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future)
    forecast.head()

    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('Forecast data')
    fig1= plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.title('Forecast components')
    fig2=m.plot_components(forecast)
    st.write(fig2)



page_names_to_funcs = {
    'Intro': intro,
    "Monte Carlo Simulator": monsim,
    'Long Short-Term Memory Predictor':ML,
    'Prophet':prop
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()