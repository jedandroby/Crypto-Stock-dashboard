import ccxt
# jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10 - this command is required to run when opening jupyter labs or ccxt wont work in jupyter. or configure a config file.
import pandas as pd
import hvplot.pandas
# from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import os
import sqlalchemy as sql
import sys
import questionary
from MCForecastTools import MCSimulation


def get_data_crypto():
    '''
    Function used for pulling data from qualified crypto exchanged based on accurate user volume taken from the free ccxt package. User chooses what exchange to connect too, and then decides what ticker they want to find. Code will find a USD equivalent pair and return the dataframe.
    '''
    # getting list of qualified exchanges for user to choose to connect too. 
    exchanges = ccxt.exchanges
    qe=['binance','bitfinex','bithumb','bitstamp','bittrex','coinbase','gemini','kraken',
    'hitbtc','huobi','okex','poloniex','yobit','zaif']
    fe=[s for s in exchanges if any(exchanges in s for exchanges in qe)]
    
    exchange_id = questionary.select("Which exchange do you wish to pull from?",                   choices=fe).ask()
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
    ohlc = exchange.fetch_ohlcv('%s/USD' % ticker, timeframe='1h', limit=691)
    # Creating a dataframe
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
    
def get_data_qqq():
    '''
    Function used for getting data from Alpaca for the QQQ Stock ticker and save the data as a dataframe.
    '''
    alpaca = tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        api_version="v2")

    # Format current date as ISO format 
    today = pd.Timestamp.now(tz="US/Pacific")
    a_year_ago = pd.Timestamp(today - pd.Timedelta(days=365)).isoformat()
    end_date = pd.Timestamp(today - pd.Timedelta(days=1)).isoformat()

    # Set the tickers
    tickers = ["QQQ"]


    # Set timeframe to one day ('1Day') for the Alpaca API
    timeframe = "1D"

    # Get current closing prices for NDX
    dqqq_price = alpaca.get_bars(
            tickers,
            timeframe,
            start=a_year_ago,
            end=end_date
        ).df

    # Display sample data
    # separate Ticker Data
    QQQ = dqqq_price[dqqq_price['symbol']=='QQQ'].drop('symbol', axis=1)
     
    # set the index as Timestamp
    # dqqq_price = dqqq_price.set_index(['Timestamp'])

    # Concatenate the Ticker DataFrames

    dqqq_price = pd.concat([QQQ],axis=1, keys=['QQQ'])
    return dqqq_price

def get_data_spy():
     '''
    Function used for getting data from Alpaca for the SPY Stock ticker and save the data as a dataframe.
    '''
    alpaca = tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        api_version="v2")
    ALPACA_API_KEY = 'PK2BWJQC7W7Z3C87TKI6'
    ALPACA_SECRET_KEY = 'qox7s7aZ70L9yAp8HS6Kz0JXngu6a20ikuD3EmCq'
    
    # # Set the variables for the Alpaca API and secret keys
    # alpaca_api_key=os.getenv(ALPACA_API_KEY)
    # # Create the Alpaca tradeapi.REST object
    # alpaca_secret_key=os.getenv(ALPACA_SECRET_KEY)
    
    alpaca=tradeapi.REST(ALPACA_API_KEY,ALPACA_SECRET_KEY,api_version="v2")

    #Setting the tickers
    tickers = ['SPY']

    #Setting the timeframe
    timeframe='1Day'

    #Formatting the date
    today = pd.Timestamp.now(tz="US/Pacific")
    a_year_ago = pd.Timestamp(today - pd.Timedelta(days=15000)).isoformat()
    end_date = pd.Timestamp(today - pd.Timedelta(days=1)).isoformat()

    #Getting the closing prices
    spy_price = alpaca.get_bars(
    tickers,
    timeframe,
    start=a_year_ago,
    end=end_date
    ).df

    # Check for NaN Values
    spy_price.isnull().dropna()

    df_spy = spy_price.drop(columns=["trade_count","vwap","symbol"])
    return df_spy
