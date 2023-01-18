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

# def get_data_crypto():
#     '''
#     Function used for pulling data from qualified crypto exchanged based on accurate user volume taken from the free ccxt package. User chooses what exchange to connect too, and then decides what ticker they want to find. Code will find a USD equivalent pair and return the dataframe.
#     '''
#     # getting list of qualified exchanges for user to choose to connect too. 
#     exchanges = ccxt.exchanges
#     qe=['binance','bitfinex','bithumb','bitstamp','bittrex','coinbase','gemini','kraken',
#     'hitbtc','huobi','okex','poloniex','yobit','zaif']
#     fe=[s for s in exchanges if any(exchanges in s for exchanges in qe)]
    
#     exchange_id = questionary.select("Which exchange do you wish to pull from?",choices=fe).ask()
#     exchange_class = getattr(ccxt, exchange_id)
#     exchange = exchange_class({
#     'timeout':30000,
#     'enableRateLimit':True,
#     })
#     # exchange
#     # load market data
#     markets=exchange.load_markets()
#     # getting open high low close data for BTC from binance us, last 1000 hour candles. 
#     # have user select ticker they want to analyze, and convert it to upper
#     ticker = str(questionary.text('Please type the ticker of the token you are trying to analyze').ask())
#     ticker=ticker.upper()
#     try:
#         ohlc = exchange.fetch_ohlcv('%s/USD' % ticker, timeframe='1h', limit=691)
#     except :
#         try:
#             ohlc = exchange.fetch_ohlcv('%s/USDC' % ticker, timeframe='1h', limit=691)
#         except:    
#             try:
#                 ohlc = exchange.fetch_ohlcv('%s/USDT' % ticker, timeframe='1h', limit=691)
#             except:
#                 print('Sorry please pick another exchange/token to analyze, could not find a USD/USDC/USDT pair.')
#                 ohlc = None
#                 pass
#     # Creating a dataframe
#     if (isinstance(ohlc,pd.DataFrame) ==True):
#         if len(ohlc) > 1:
#             df = pd.DataFrame(ohlc,columns=['timestamp','Open','High','Low','Close','Volume'])
#             # Check for null values
#             df.isnull().sum().dropna()
#             # taken unix to datetime from firas pandas extra demo code
#             def unix_to_date(unix):
#                 return pd.to_datetime(unix, unit = "ms").tz_localize('UTC').tz_convert('US/Pacific')
#             # Clean unix timestamp to a human readable timestamp. 
#             df['timestamp']= df['timestamp'].apply(unix_to_date)
#             # set index as timestamp
#             df = df.set_index(['timestamp'])
#             return df
#     return None
# df = get_data_crypto()

def analyze_data(data):
        d = data
        # # get the percent change for the coin and drop NaN values
        coin_pct_change = d['Close'].pct_change().dropna()
        # coin_pct_change = pd.DataFrame(coin_pct_change)
        coin_annual_pct_change = coin_pct_change.mean() * 365

        # calculate the annual std for BTC
        coin_annual_std = coin_pct_change.std() * (365) ** (1/2)


        # calculate the variance for the coin
        coin_variance = coin_pct_change.var()
        st.subheader("Asset Analysis")

        st.write(f" The Variance is: {coin_variance: .5f}")
        st.caption("""The Variance measures the deviation of the asset from the average (mean) price. A higher number will generally 
        indicate a more volitile asset, as it tends to deviate from the mean price more consistently. But, this could also mean that 
        you have the ability to make more return on your investment, as there is more deviation from the average price. A lower number
        demonstrates the asset is less volitile, and could be seen as a 'safer' investment.""")
        # calculate the sharpe ratio for the coin
        sharpe_ratio = coin_annual_pct_change / coin_annual_std
        st.write(f" The Sharpe Ratio is: {sharpe_ratio: .2f}")
        st.caption(""" The Sharpe Ratio takes the assets average annual return and divides it by the assets annual standard deviation.
        This is used to measure the risk/reward that you would be taking in a trade. Generally, a Sharpe Ratio between 1 - 2 is 
        considered good, and anything over 3 is amazing!""")

        # # calculate the covariance between the coin and SPY
        # cov = coin_pct_change.cov(d['Close'].pct_change())
        # st.write(f" The covariance to SPY is: {cov: .2f}")

        # calculate and pring the mean cumulative returns for the coin
        # get the annual pct change for the coin
        coin_annual_pct_change = coin_annual_pct_change * 100
        st.write(f" The Annual Percent Return is: % {coin_annual_pct_change: .2f}")
        st.caption("""The Annual Percent Return demonstrates the annual rate of return for the asset. The Higher rate of return, the better! 
        To calculate this, we take the average returns of the asset, and multiply it by 365, which gives us the average annual return!""")

            # create and plot the SMA for a 50 and 200 day period
        ax = d['Close'].plot(figsize=(10,7),title= 'Asset chart with 50 and 200 Simple Moving Average',ylabel='Price in USD $',xlabel='Time since first data point' )
        d['Close'].rolling(window=200).mean().plot(ax=ax)
        d['Close'].rolling(window=50).mean().plot(ax=ax, color= 'Red')
        ax.legend(["Daily Prices", "50-Day Rolling Average", '200 day rolling average'])
        
# analyze_data(df)



def monte_carlo_sim(data):   
        d = data
        #Next, we calculate the number of days that have elapsed in our chosen time window
        time_elapsed = len(d)


        #Current price / first record (e.g. price at beginning of 2009)
        #provides us with the total growth %
        total_growth = (d['Close'].iloc[-1] / d['Close'].iloc[0])

        #Next, we want to annualize this percentage
        #First, we convert our time elapsed to the # of years elapsed
        number_of_years = time_elapsed / 365.0
        #Second, we can raise the total growth to the inverse of the # of years
        #(e.g. ~1/10 at time of writing) to annualize our growth rate
        cagr = total_growth ** (1/number_of_years) - 1

        #Now that we have the mean annual growth rate above,
        #we'll also need to calculate the standard deviation of the
        #daily price changes
        std_dev = d['Close'].pct_change().std()

        #Next, because there are 365 trading days in a year for Crypto,
        #we'll need to scale this by an annualization factor
        #reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

        number_of_trading_days = 365
        std_dev = std_dev * math.sqrt(number_of_trading_days)

        #From here, we have our two inputs needed to generate random
        #values in our simulation
        # st.write("Compound Annual Growth Rate (cagr): ", str(round(cagr,4)))
        # st.caption("""The cagr is used to measure the compounded growth of an asset over a yearly basis. In this case, we are measuring the annual 
        # compounded returns, so you can think of this as the average compounded annual return""")
        st.write("Standard Deviation (std)", str(round(std_dev,2)))
        st.caption(""" The Standard Deviation can be used as a volitlity metric, and showcase the spread in which an asset deviates from the 
        average price. Standard Deviation is more a measure of how far apart numbers are from each other, whereas
        the variance will return a value to show how much the numbers vary from the mean. Generally, a std value over 1, will be considered
        more volitile, whereas a std under 1 is seen as less volitile. """)


        st.subheader("50 and 200 Day Simple Moving Average Chart")

        #Generate random values for 1 year's worth of trading (365 days),
        #using numpy and assuming a normal distribution
        daily_return_percentages =  np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1

        #Now that we have created a random series of future
        #daily return %s, we can simply apply these forward-looking
        #to our last stock price in the window, effectively carrying forward
        #a price prediction for the next year

        #This distribution is known as a 'random walk'

        price_series = [d['Close'].iloc[-1]]

        for j in daily_return_percentages:
            price_series.append(price_series[0] * j)

        #Great, now we can plot of single 'random walk' of stock prices
        # plt.plot(price_series)

        plot_0_5 = plt.show()
        st.pyplot(plot_0_5)

        #Now that we've created a single random walk above,
        #we can simulate this process over a large sample size to
        #get a better sense of the true expected distribution
        number_of_trials = 500
        st.subheader("One Year Monte Carlo Simulation - 500 trials")
        #set up an additional array to collect all possible
        #closing prices in last day of window.
        #We can toss this into a histogram
        #to get a clearer sense of possible outcomes
        closing_prices = []

        for i in range(number_of_trials):
            #calculate randomized return percentages following our normal distribution
            #and using the mean / std dev we calculated above
            daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
            price_series = [d['Close'].iloc[-1]]

            for j in daily_return_percentages:
                #extrapolate price out for next year
                price_series.append(price_series[-1] * j)

            #append closing prices in last day of window for histogram

            closing_prices.append(price_series[-1])

            #plot all random walks
            plot_please = plt.plot(price_series)


        plot_please = plt.show()
        st.pyplot(plot_please)

        #lastly, we can split the distribution into percentiles
        #to help us gauge risk vs. reward
        st.subheader("Distribution Chart for simulation results")
        #Pull top 10% of possible outcomes
        top_ten = np.percentile(closing_prices,100-10)

        #Pull bottom 10% of possible outcomes
        bottom_ten = np.percentile(closing_prices,10);
        # bins=40
        #create histogram again
        plt.hist(closing_prices)
        #append w/ top 10% line
        plt.axvline(top_ten,color='r',linestyle='dashed',linewidth=2)
        #append w/ bottom 10% line
        plt.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)
        #append with current price
        plt.axvline(d['Close'].iloc[-1],color='g', linestyle='dashed',linewidth=2)

        plot_3 = plt.show()
        st.pyplot(plot_3)
        #from here, we can check the mean of all ending prices
        #allowing us to arrive at the most probable ending point 
        st.subheader("Monte Carlo Price Expectation Results")
        mean_end_price = round(np.mean(closing_prices),2)
        st.write("The Expected price of the asset in one year is : $", str(mean_end_price))
        st.caption("This is calculated by taking the average (mean) closing price of all the simulations.")

    


def get_data_yahoo (START, TODAY):
    stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
    selected_ticker = st.selectbox("Pick a coin for prediction",stocks)
    @st.cache
    def load_data(selected_ticker):
        data = yf.download(selected_ticker,START,TODAY)
        data.reset_index('Date',inplace=True)
        data = pd.DataFrame(data)
        return data
    data = load_data(selected_ticker)
    st.subheader("Asset Data Head")
    st.write(data.head())
    st.subheader('Asset Data Tail')
    st.write(data.tail())
    st.subheader("Interactive Asset Chart")
    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"],y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data["Date"],y=data['Close'], name='stock_close'))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()
    return data

    
def LSTM_model(look_back, neurons, epoch, data, test_split):
       
    # create a time series dataset
    def create_dataset(prices, look_back=3):
        X, y = [], []
        for i in range(len(prices)-look_back-1):
            X.append(prices[i:(i+look_back), 0])
            y.append(prices[i + look_back, 0])
        return np.array(X), np.array(y)
    # Forecasting
    # extract the close prices
    prices = data['Close'].values

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = scaler.fit_transform(prices.reshape(-1, 1))
    
    # create x,y data sets
    X, y = create_dataset(prices, look_back)
    
    # reshape the input to be 3D [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # split the data into train and test sets
    train_size = int(len(X) * test_split)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    # get the data rdy to plot for later
    close_data = prices.reshape((-1,1))

    split_percent = test_split
    split = int(split_percent*len(close_data))

    date_test = data['Date'][split:]
    
    # create the LSTM model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense((neurons/2), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
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

    # Create a list of dates for the next x days
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
    layout2 = go.Layout(
        title = "Real price + Model future price predictions ",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig= go.Figure(data=[trace2, trace1], layout=layout2)
    st.plotly_chart(fig)
    expected_price=y_future.iloc[-1,-1]
    # expected_price=expected_price.values
    st.write(f'in {look_back} days the model predicts the price to be around ${expected_price}')
    