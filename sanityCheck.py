from portfolio.get_data import get_data
from portfolio.assess_portfolio import get_portfolio_value, get_portfolio_stats
print("Imports OK")

import pandas as pd

""""
dates = pd.date_range("2011-01-03", "2011-01-31")
prices = get_data(["AAPL","SPY"], dates, path="data-2011")
print(prices.head())
"""


orders_df = pd.read_csv("orders/orders-short.csv", index_col="Date", parse_dates=True).sort_index()
start_date, end_date = orders_df.index.min(), orders_df.index.max()
symbols = orders_df["Symbol"].unique().tolist()
print(start_date.date(), end_date.date(), symbols)