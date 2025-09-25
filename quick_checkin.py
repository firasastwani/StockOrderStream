# compute the number of trading days using SPY first

import pandas as pd

# Read the SPY data from the data-2011 folder
spy_df = pd.read_csv('data-2011/SPY.csv')

# Count the number of trading days (number of rows, excluding header)
num_trading_days = len(spy_df)

print(f"Number of trading days (SPY): {num_trading_days}")

print()

orders_files = ['orders/orders-short.csv', 'orders/orders.csv', 'orders/orders2.csv']

# Iterate over each orders file and compute the number of trading days from first to last trade
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_trading_days = set(spy_df['Date'])

for orders_file in orders_files:
    
    orders_df = pd.read_csv(orders_file)
    orders_df['Date'] = pd.to_datetime(orders_df['Date'])

    # Get the range from first trade to last trade
    first_trade_date = orders_df['Date'].min()
    last_trade_date = orders_df['Date'].max()
    
    # Count SPY trading days within this range
    spy_days_in_range = spy_df[(spy_df['Date'] >= first_trade_date) & (spy_df['Date'] <= last_trade_date)]
    num_trading_days_in_range = len(spy_days_in_range)
    
    print(f"{orders_file}")
    print(f"First trade date: {first_trade_date.strftime('%Y-%m-%d')}")
    print(f"Last trade date: {last_trade_date.strftime('%Y-%m-%d')}")
    print(f"Number of SPY trading days from first to last trade: {num_trading_days_in_range}\n")

