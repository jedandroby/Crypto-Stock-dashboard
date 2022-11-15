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
from warnings import filterwarnings
filterwarnings("ignore")



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
        ohlc = exchange.fetch_ohlcv('%s/USD' % ticker, timeframe='1h', limit=691)
    except :
        try:
            ohlc = exchange.fetch_ohlcv('%s/USDC' % ticker, timeframe='1h', limit=691)
        except:
            try:
                ohlc = exchange.fetch_ohlcv('%s/USDT' % ticker, timeframe='1h', limit=691)
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




def analyze_data():
      
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
    print(f" The Variance for BTC is: {coin_variance: .2f}")
    
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

analyze_data()
# # connect to database function
# def connect():
#     '''Connect to the sqlite database server''' 
#     conn = None
#     try:
#         # connect to the SQLite server
#         print('Connecting to the SQLite database...')
#         conn = sql.create_engine('sqlite:///')
#     except :
#         print("Connection not successful!")
#         sys.exit(1)
#     print("Connection Successful!")
#     return conn
# #copy data to database
# def copy_to_db(conn, df, table):
#     """
#     save the dataframe in memory as a sqlite database with name 'table' 
#     conn is the connection engine used. use connect function to get 
#     """    
#     try:
#         df.to_sql('%s'%table, con=conn, index=True,if_exists='replace')
#         print(f'Saving dataframe as a table called {table} in sqlite database')
#     except :
#         print("Error")
#     print("Done!")
#     return conn.table_names()
# #copy a dataframe from database based on a query
# def open_as_df(query,conn):
#     '''pass query to get dataframe: select * from spy_db_OHLCV fx. '''
#     try:
#         df = pd.read_sql_query(sql = query,con = conn, index_col= ['timestamp'])
#         print('Accessing SQLite database based on query')
#     except :
#         print('Error')
#         sys.exit(1)
#     return df