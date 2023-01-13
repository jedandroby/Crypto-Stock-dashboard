
import streamlit as st
from datetime import date
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly






def monsim():
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Monte Carlo Asset Predictor")

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

    # data_load_state = st.text("Load data")
    data = load_data(selected_stocks)
    # data_load_state.text("Loading data")

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
    # Forecasting
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

    analyze_data(data)



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


    monte_carlo_sim(data)


    st.write("Disclaimer")
    st.caption("""The values that are displayed in this dashboard are solely there for the purpose of knowledge and education. This in no
        way is financial advice, and we strongly recommend to take into account many other factors before entering a trade. With that being said,
        we hope you found this information helpful, and we wish you the best of luck on your trading endeavours!""")

def intro ():
    st.title("Hi")
    st.write('Welcome to our advanced financial platform, designed to provide you with the tools and insights you need to make informed investment decisions. Our platform combines cutting-edge predictive models, such as Monte Carlo simulations, machine learning, and algorithmic trading, with a wealth of historical market data, to provide unparalleled insights into the performance of a wide range of assets.'
    "Our advanced models include a Monte Carlo asset predictor, a time series predictor, a backtesting feature for your trading indicators, and a logistic regression model. These tools allow you to test and optimize your investment strategies, as well as gain a deeper understanding of the underlying factors that affect asset prices."
    "Whether you're a professional trader, a seasoned investor, or just starting out, our platform can help you make better-informed decisions. By providing you with the latest predictive tools and a wealth of historical data, our platform can give you the edge you need to succeed in today's fast-paced financial markets."
    "Experience the difference that advanced predictive tools can make in your financial success. Sign up for our platform today and gain access to the insights and tools you need to make informed investment decisions.")
    
def ML ():
    st.write('did this work?')

def prop():
# Forecasting
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Prophet Asset Predictor")

    stocks = ("BTC-USD","LINK-USD","SOL-USD","MATIC-USD","MANA-USD","DOT-USD","AVAX-USD","XLM-USD","LTC-USD","XRP-USD","BNB-USD","UNI-USD","ETH-USD","ADA-USD","USDC-USD","BAT-USD")
    selected_stocks = st.selectbox("Pick a coin for prediction",stocks)

    n_days = st.slider("Days of Prediction:",1,7)
    period = n_days * 365


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker,START,TODAY)
        data.reset_index('Date',inplace=True)
        data = pd.DataFrame(data)
        return data

    # data_load_state = st.text("Load data")
    data = load_data(selected_stocks)
    # data_load_state.text("Loading data")
    st.subheader('Raw Data')
    st.write(data.tail())
    
    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"],y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data["Date"],y=data['Close'], name='stock_close'))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data() 




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

# def Prophet ():

page_names_to_funcs = {
    'Intro': intro,
    "Monte Carlo Simulator": monsim,
    'Machine learning algorithms':ML,
    'Prophet':prop
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()