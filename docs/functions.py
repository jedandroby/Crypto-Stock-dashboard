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
from finta import TA
from sklearn.linear_model import LinearRegression

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


        # plot_1 = plt.show()
        # st.line_chart(plot_1)

        # #plot histogram
        # st.bar_chart(closing_prices)

        # # ,bins=40
        # plot_2 = plt.show()
        # st.pyplot(plot_2)

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

    


def get_data_yahoo ():
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
    selected_ticker = st.selectbox("Pick a coin for prediction",stocks)
    @st.cache(allow_output_mutation=True)
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
    
    
def trading_algo(data):
    # Space out the maps so the first one is 2x the size of the other three
    # col1, col2 = st.columns(2)
    # with col1:
    def execute_trade(data, strategy,rsi_low,rsi_high):
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
                        if rsi[i] < rsi_low:
                            buy_indices.append(i)
                            buy_closes.append(data.loc[i, 'Close'])

                        elif rsi[i] > rsi_high:
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
    def execute_trades(data, strategy,fastperiod,slowperiod,signalperiod):
                # Initialize indicators

                exp1 = data["Close"].ewm(fastperiod, adjust=False).mean()
                exp2 = data['Close'].ewm(slowperiod, adjust=False).mean()
                macd = exp1 - exp2
                macd_df = pd.DataFrame(macd, columns=['MACD'])
                macd_signal = macd.ewm(signalperiod, adjust=False).mean()

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
    st.header("Trading Algorithm Automatic Backtester")
    st.write("""Welcome to the Automatic Trading Algorithm Backtester! 
    Say goodbye to manual backtesting and hello to efficient and accurate results.
    With our cutting-edge technology, you can easily test and optimize your trading 
    strategies, all with just a few clicks. Whether you're a seasoned trader or just starting out, 
    our backtester is the perfect tool to help you make informed decisions and maximize your returns. 
    Get ready to revolutionize the way you trade and experience the power of automated backtesting. Join 
    us now and start seeing the results you've always wanted!""")
    
    def plotting(strategy,rsi_low,rsi_high,fastperiod,slowperiod,signalperiod):
        if strategy == "RSI":
            execute_trade(data,selected_strategy, rsi_low,rsi_high)
        else:            
            execute_trades(data,strategy,fastperiod,slowperiod,signalperiod)   

    # with col2:
        
    st.header("Pick A Trading Strategy")  
    trading_meth = ("RSI",'MACD')
    selected_strategy = st.selectbox("strategies",trading_meth)

    strategy = (selected_strategy)
    if strategy == "RSI":
        st.sidebar.title("What is the RSI Strategy")
        st.sidebar.write("""The RSI strategy is based on the Relative Strength Index, which is a technical indicator that compares the magnitude of recent gains to recent losses in order to assess overbought or oversold conditions of an asset. In this strategy, when the RSI falls below a user-specified low level, it generates a buy signal, 
        and when the RSI rises above a user-specified high level, it generates a sell signal. You may adjust the low and high levels to suit your own risk tolerance and investment style. A general RSI trading strategy will have the overbought value as 30 and oversold value as 70. There are many other ways to configure the strategy as well, so feel free to try out new values for the overbought and oversold!""")
        st.sidebar.title("Get started below!")
        st.sidebar.subheader("Customize The RSI Strategy")
        share_size = st.sidebar.slider("How many shares would you like to purchase on every buy signal?",1,100)
        st.sidebar.caption("Share size refers to the number of shares the algorithm will purchase on every buy signal.")
        rsi_low = st.sidebar.slider("What would you like the low signal for the RSI to be?",10,40)
        st.sidebar.caption("Low signal for RSI refers to the RSI threshold at which the algorithm will generate a buy signal.")
        rsi_high = st.sidebar.slider("What would you like the high signal for the RSI to be?",60,100)
        st.sidebar.caption("High signal for RSI refers to the RSI threshold at which the algorithm will generate a sell signal.")
        if st.sidebar.button("Start"):
            st.snow()
            # code to run after submit button is clicked
            execute_trade(data,selected_strategy,rsi_low,rsi_high)
            def stats(strategy):

                if strategy == "RSI":

                    def backtest_RSI(data, share_size, rsi_low, rsi_high):
                        # initial_capital = int(initial_capital)
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
                                if rsi[i] < rsi_low:
                                    buy_indices.append(data.loc[i, 'Date'])
                                    buy_closes.append(data.loc[i, 'Close'])

                                elif rsi[i] > rsi_high:
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
                        df2['Total Cost'] = 0.0
                        df2['Total Revenue'] = 0.0
                        # initial_capital = int(initial_capital)
                        share_size = int(share_size)

                        for i in range(0, len(df2)):
                            # buy if RSI < 30
                            if df2.loc[i, 'Signal'] == 'Buy': 
                                # calculate the cost of the trade
                                cost = -(df2.loc[i, 'Price'] * share_size)
                                df2.loc[i, 'Total Cost'] += cost  
                            # sell if RSI > 70
                            elif df2.loc[i, 'Signal'] == 'Sell': 
                                # Sell condition
                                # calculate the proceeds of the trade
                                revenue = (df2.loc[i, 'Price'] * share_size)
                                df2.loc[i, 'Total Revenue'] += revenue                   
                            # no signal if RSI between 30 and 70
                            else:
                                skip
                        st.title("Trading Algorithm Results") 
                        total_revenue = df2['Total Revenue'].sum()
                        total_cost = df2['Total Cost'].sum()
                        st.subheader("Buy and Sell Signal DataFrame")
                        st.write(df2)
                        roi = round(total_revenue / -(total_cost), 2)
                        total_cost = total_cost * -1
                        st.write(f"The trading algorithm resulted in a return on investment of {round(roi, 2)}%")
                        st.write(f"The total money invested was ${round(total_cost, 2)}")
                        st.write(f"The total Revenue generated was ${round(total_revenue, 2)}")
                        usd_val = total_revenue - total_cost
                        st.write(f"""The total amount in USD that you would have made while trading this 
                        strategy would have been: ${round(usd_val, 2)}""")
                        st.caption('''Please note that this trading algorithm is a generic strategy 
                        using pre-set conditions and is intended for educational purposes only. In 
                        order to achieve a higher return on investment, it is recommended to tweak 
                        the conditions and customize the strategy to suit your needs. Keep in mind 
                        that past performance is not indicative of future results and this is not 
                        financial advice. But with our subscription model, we will work together to 
                        further amplify the strategy and create even greater returns on investment!''')
                backtest_RSI(data, share_size,rsi_low, rsi_high)
            stats(strategy)

    else:
        st.sidebar.title("What is the MACD Strategy?")
        st.sidebar.write("""The Moving Average Convergence Divergence MACD strategy uses the difference between two moving averages, the fast Exponential Moving Average (EMA) and slow EMA, to generate buy and sell 
            signals. When the fast EMA crosses above the slow EMA, it generates a buy signal, indicating that the asset's price is likely to rise. On 
            the other hand, when the fast EMA crosses below the slow EMA, it generates a sell signal, indicating that the stock's price is likely to fall. 
            You may adjust the EMA periods to your liking to fine-tune the strategy to fit your specific needs. A baseline that traders generally use, would be to have the slow period EMA be 26, the fast be 12, and the signal to be 9. This tends to be a good baseline, 
            but there are many other configurations that you could try as well! """)
        st.sidebar.title("Get started below!")
        st.sidebar.subheader("Customize The MACD Strategy")
        share_size = st.sidebar.slider("How many shares would you like to purchase on every buy signal?",1,1000)
        st.sidebar.caption("Share size refers to the number of shares the algorithm will purchase on every buy signal.")
        fastperiod = st.sidebar.slider("What would you like the fast (shorter) period EMA to be?(Normal is 12)",4,20)
        st.sidebar.caption("Fast period for EMA refers to the shorter time period used to calculate the Exponential Moving Average.")
        slowperiod = st.sidebar.slider("What would you like the slow (longer) period EMA to be?(Normal is 26)",21,40)
        st.sidebar.caption("Slow period for EMA refers to the longer time period used to calculate the Exponential Moving Average.")
        signalperiod = st.sidebar.slider("What would you like the signal period EMA to be?(Normal is 9)",3,20)
        st.sidebar.caption("Signal period for EMA refers to the time period used to calculate the signal line for the Exponential Moving Average.")
        if st.sidebar.button("Start"):
            st.snow()
            # code to run after submit button is clicked
            # plotting(strategy,rsi_low,rsi_high,fastperiod,slowperiod,signalperiod)
            execute_trades(data,strategy,fastperiod,slowperiod,signalperiod)  
        def stats(strategy):
            if strategy == "MACD":

                def backtest_MACD(data, share_size,fastperiod, slowperiod, signalperiod):
                    # initial_capital = int(initial_capital)
                    share_size = int(share_size)
                    # Initialize variables to keep track of trades
                    buy_indices = []
                    buy_closes = []
                    sell_indices = []
                    sell_closes = []
                    exp1 = data["Close"].ewm(fastperiod, adjust=False).mean()
                    exp2 = data['Close'].ewm(slowperiod, adjust=False).mean()
                    macd = exp1 - exp2
                    macd_df = pd.DataFrame(macd, columns=['MACD'])
                    macd_signal = macd.ewm(signalperiod, adjust=False).mean()
                    # Iterate through the data and execute trades
                    for i in range(1, len(data)):
                        if strategy == 'MACD':
                            if macd[i] < macd_signal[i]:
                                buy_indices.append(data.loc[i, 'Date'])
                                buy_closes.append(data.loc[i, 'Close'])
                            elif macd[i] > macd_signal[i]:
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
                    df2['Total Cost'] = 0.0
                    df2['Total Revenue'] = 0.0
                    # initial_capital = int(initial_capital)
                    share_size = int(share_size)

                    for i in range(0, len(df2)):
                        # buy if RSI < 30
                        if df2.loc[i, 'Signal'] == 'Buy': 
                            # Buy condition
                            # calculate the cost of the trade
                            cost = -(df2.loc[i, 'Price'] * share_size)
                            df2.loc[i, 'Total Cost'] += cost
                            # add the number of shares purchased to the accumulated shares
                            # accumulated_shares += share_size

                        # sell if RSI > 70
                        elif df2.loc[i, 'Signal'] == 'Sell': 
                            # Sell condition
                            # calculate the proceeds of the trade
                            revenue = (df2.loc[i, 'Price'] * share_size)
                            df2.loc[i, 'Total Revenue'] += revenue                   
                            # no signal if RSI between 30 and 70
                        else:
                            skip
                    st.subheader("Trading Algorithm Results")        
                    total_revenue = df2['Total Revenue'].sum()
                    total_cost = df2['Total Cost'].sum()
                    st.subheader("Buy and Sell Signal DataFrame")
                    st.write(df2)
                    roi = round(total_revenue / -(total_cost), 2)
                    total_cost = total_cost * -1
                    st.write(f"The trading algorithm resulted in a return on investment of {round(roi, 2)}%")
                    st.write(f"The total money invested was ${round(total_cost, 2)}")
                    st.write(f"The total Revenue generated was ${round(total_revenue, 2)}")
                    usd_val = total_revenue - total_cost
                    st.write(f"""The total amount in USD that you would have made while trading this 
                    strategy would have been: ${round(usd_val, 2)}""")
                    st.caption('''Please note that this trading algorithm is a generic strategy 
                    using pre-set conditions and is intended for educational purposes only. In 
                    order to achieve a higher return on investment, it is recommended to tweak 
                    the conditions and customize the strategy to suit your needs. Keep in mind 
                    that past performance is not indicative of future results and this is not 
                    financial advice. But with our subscription model, we will work together to 
                    further amplify the strategy and create even greater returns on investment!''')
                backtest_MACD(data, share_size,fastperiod, slowperiod, signalperiod)


        stats(strategy)

        
def linear_model(df):
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
    
    st.subheader('Actual Returns vs Strategy Returns')
    if st.button('Click to see Dataframe'):
        st.write(df.head(10))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=x, name='Actual Returns', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=y, name='Strategy Returns', line=dict(color='red')))
    fig.update_layout(title='Comparison of Actual Returns and Strategy Returns',xaxis_title='Date',yaxis_title='Returns')
    if st.button('Click here to see the plot'):
        st.plotly_chart(fig)

        
def prop_model(data, period):
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
    