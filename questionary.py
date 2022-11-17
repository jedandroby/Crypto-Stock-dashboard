import ccxt
# jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10 - this command is required to run when opening jupyter labs or ccxt wont work in jupyter. or configure a config file.
import pandas as pd
import os
# import sqlalchemy as sql
import sys
import questionary
from MCForecastTools import MCSimulation
from warnings import filterwarnings
filterwarnings("ignore")
import pandas as pd
import hvplot.pandas
import numpy as np
from MCForecastTools import MCSimulation
from warnings import filterwarnings
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import questionary


def get_data_crypto():
    '''
    Function used for pulling data from qualified crypto exchanged based on accurate user volume taken from the free ccxt package. User chooses what exchange to connect too, and then decides what ticker they want to find. Code will find a USD equivalent pair and return the dataframe.
    '''
    # getting list of qualified exchanges for user to choose to connect too.
    exchanges = ccxt.exchanges
    qe=['binance','bitfinex','bithumb','bitstamp','bittrex','coinbase','gemini','kraken',
    'hitbtc','huobi','okex','poloniex','yobit','zaif']
    fe=[s for s in exchanges if any(exchanges in s for exchanges in qe)]
    exchange_id = questionary.select("Which exchange do you wish to pull from?",choices=fe).ask()
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
    'timeout':30000,
    'enableRateLimit':True,
    })
    # exchange
    # load market data
    markets=exchange.load_markets()
    # getting open high low close data for BTC from binance us, last 1000 hour candles.
    # have user select ticker they want to analyze, and convert it to upper
    ticker = str(questionary.text('Please type the ticker of the token you are trying to analyze').ask())
    ticker=ticker.upper()
    try:
        ohlc = exchange.fetch_ohlcv('%s/USD' % ticker, timeframe='1d', limit=2000)
    except :
        try:
            ohlc = exchange.fetch_ohlcv('%s/USDC' % ticker, timeframe='1d', limit=2000)
        except:
            try:
                ohlc = exchange.fetch_ohlcv('%s/USDT' % ticker, timeframe='1d', limit=2000)
            except:
                print('Sorry please pick another exchange/token to analyze, could not find a USD/USDC/USDT pair.')
                ohlc = None
                pass
    # Creating a dataframe
    # if (isinstance(ohlc,pd.DataFrame) ==True):
    #     if len(ohlc) > 1:
    df = pd.DataFrame(ohlc,columns=['timestamp','Open','High','Low','Close','Volume'])
    # Check for null values
    df.isnull().sum().dropna()
    # taken unix to datetime from firas pandas extra demo code
    def unix_to_date(unix):
        return pd.to_datetime(unix, unit = "ms").tz_localize('UTC').tz_convert('US/Pacific')
    # Clean unix timestamp to a human readable timestamp.
    df['timestamp']= df['timestamp'].apply(unix_to_date)
    # set index as timestamp
    df = df.set_index(['timestamp'])
    return df
    # return None
d = get_data_crypto()
print(d)




def analyze_data(d):
    d = d
    # # get the percent change for the coin and drop NaN values
    coin_pct_change = d['Close'].pct_change().dropna()
  
    # get the annual pct change for the coin
    coin_annual_pct_change = coin_pct_change.mean() * 365
   
    # calculate the annual std for BTC
    coin_annual_std = coin_pct_change.std() * (365) ** (1/2)
  
    # create and plot the SMA for a 50 and 200 day period
    ax = d['Close'].plot(figsize=(10,7), title="Daily prices versus 180-Day and 7-day Rolling Average")
    d['Close'].rolling(window=200).mean().plot(ax=ax)
    d['Close'].rolling(window=50).mean().plot(ax=ax, color= 'red')
    ax.legend(["Daily Prices", "50-Day Rolling Average", '200 day rolling average'])
   
    # calculate the variance for the coin
    coin_variance = coin_pct_change.var()
    print(f" The Variance is: {coin_variance: .6f}")
    
    # calculate the sharpe ratio for the coin
    sharpe_ratio = coin_annual_pct_change / coin_annual_std
    print(f" The Sharpe Ratio is: {sharpe_ratio: .2f}")
    
    # calculate the covariance between the coin and SPY
    cov = d['Close'].pct_change().cov(d['Close'].pct_change())*100000
    print(f" The covariance to SPY is: {cov: .2f}")
    
    # calculate and pring the mean cumulative returns for the coin
    cum_returns = (1 + coin_pct_change).cumprod() - 1 
    cum_returns_mean = cum_returns.mean()
    print(f' The average Cumulative Return is: % {cum_returns_mean: .2f}')

analyze_data(d)



def monte_carlo_sim(d):   
    d = d
    #Next, we calculate the number of days that have elapsed in our chosen time window
    time_elapsed = (d.index[-1] - d.index[0]).days

    #Current price / first record (e.g. price at beginning of 2009)
    #provides us with the total growth %
    total_growth = (d['Close'][-1] / d['Close'][1])

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

    #Next, because there are roughy ~252 trading days in a year,
    #we'll need to scale this by an annualization factor
    #reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

    number_of_trading_days = 252
    std_dev = std_dev * math.sqrt(number_of_trading_days)

    #From here, we have our two inputs needed to generate random
    #values in our simulation
    print ("cagr (mean returns) : ", str(round(cagr,4)))
    print ("std_dev (standard deviation of return : )", str(round(std_dev,4)))

    #Generate random values for 1 year's worth of trading (252 days),
    #using numpy and assuming a normal distribution
    daily_return_percentages =  np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1

    #Now that we have created a random series of future
    #daily return %s, we can simply apply these forward-looking
    #to our last stock price in the window, effectively carrying forward
    #a price prediction for the next year

    #This distribution is known as a 'random walk'

    price_series = [d['Close'][-1]]

    for j in daily_return_percentages:
        price_series.append(price_series[-1] * j)

    #Great, now we can plot of single 'random walk' of stock prices
    plt.plot(price_series)
    plt.show()

    #Now that we've created a single random walk above,
    #we can simulate this process over a large sample size to
    #get a better sense of the true expected distribution
    number_of_trials = 3000

    #set up an additional array to collect all possible
    #closing prices in last day of window.
    #We can toss this into a histogram
    #to get a clearer sense of possible outcomes
    closing_prices = []

    for i in range(number_of_trials):
        #calculate randomized return percentages following our normal distribution
        #and using the mean / std dev we calculated above
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
        price_series = [d['Close'][-1]]

        for j in daily_return_percentages:
            #extrapolate price out for next year
            price_series.append(price_series[-1] * j)

        #append closing prices in last day of window for histogram
        closing_prices.append(price_series[-1])

        #plot all random walks
        plt.plot(price_series)

    plt.show()

    #plot histogram
    plt.hist(closing_prices,bins=40)

    plt.show()

    #lastly, we can split the distribution into percentiles
    #to help us gauge risk vs. reward

    #Pull top 10% of possible outcomes
    top_ten = np.percentile(closing_prices,100-10)

    #Pull bottom 10% of possible outcomes
    bottom_ten = np.percentile(closing_prices,10);

    #create histogram again
    plt.hist(closing_prices,bins=40)
    #append w/ top 10% line
    plt.axvline(top_ten,color='r',linestyle='dashed',linewidth=2)
    #append w/ bottom 10% line
    plt.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)
    #append with current price
    plt.axvline(d['Close'][-1],color='g', linestyle='dashed',linewidth=2)

    plt.show()

    #from here, we can check the mean of all ending prices
    #allowing us to arrive at the most probable ending point
    mean_end_price = round(np.mean(closing_prices),2)
    print("Expected price: ", str(mean_end_price))

monte_carlo_sim(d)
