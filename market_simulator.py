import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import sys


def _load_price_series(symbol: str, data_dir: str) -> pd.Series:
    """Load Close price series for a symbol from data_dir, indexed by Date."""
    csv_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price file not found for symbol {symbol}: {csv_path}")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    if 'Close' not in df.columns:
        # Try lowercase or adjusted naming fallbacks
        for col in ['close', 'Adj Close', 'adj_close', 'AdjClose']:
            if col in df.columns:
                df = df.rename(columns={col: 'Close'})
                break
    return df['Close']


def _build_trading_calendar(spy_prices: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Return SPY trading dates between first and last order inclusive."""
    spy_window = spy_prices.loc[(spy_prices.index >= start_date) & (spy_prices.index <= end_date)]
    return spy_window.index


def compute_portvals(orders_file: str = "./orders/orders.csv", start_val: float = 1000000, args=None) -> pd.Series:
    """
    Compute daily portfolio value from orders, priced by CSVs in data_dir.

    Orders CSV format: Date,Symbol,Order,Shares
    Prices are taken from data_dir/<SYMBOL>.csv using the Close column.
    Commission and market impact are applied per order.
    """

    # Read config
    data_dir = getattr(args, 'data_dir', 'data-2011') if args is not None else 'data-2011'
    commission = getattr(args, 'transaction_fee', 9.95) if args is not None else 9.95
    market_impact_factor = getattr(args, 'market_impact_factor', 0.005) if args is not None else 0.005
    verbose = getattr(args, 'do_verbose', False) if args is not None else False

    # Read orders
    orders_df = pd.read_csv(orders_file)
    orders_df['Date'] = pd.to_datetime(orders_df['Date'])
    orders_df = orders_df.sort_values('Date')
    if orders_df.empty:
        return pd.Series(dtype=float)

    first_trade_date = orders_df['Date'].min()
    last_trade_date = orders_df['Date'].max()

    # Load SPY prices to define trading calendar
    spy_prices = _load_price_series('SPY', data_dir)
    trading_dates = _build_trading_calendar(spy_prices, first_trade_date, last_trade_date)
    if len(trading_dates) == 0:
        return pd.Series(dtype=float)

    # Load symbol price series and align to trading calendar with forward fill
    symbols = sorted(orders_df['Symbol'].unique().tolist())
    symbol_to_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            series = _load_price_series(sym, data_dir)
            series = series.reindex(trading_dates).ffill()
            symbol_to_prices[sym] = series
        except FileNotFoundError:
            # Skip symbols without data; they will be ignored
            if verbose:
                print(f"Warning: Missing data for {sym}, skipping its orders")

    # Initialize portfolio
    cash = float(start_val)
    holdings = {sym: 0 for sym in symbols}
    portvals: list[float] = []

    # Process each trading day
    orders_by_date = {d: df for d, df in orders_df.groupby('Date')}
    for current_date in trading_dates:
        if current_date in orders_by_date:
            daily_orders = orders_by_date[current_date]
            for _, order in daily_orders.iterrows():
                sym = order['Symbol']
                order_type = str(order['Order']).upper()
                shares = int(order['Shares'])

                if sym not in symbol_to_prices:
                    continue
                price = symbol_to_prices[sym].loc[current_date]
                if np.isnan(price):
                    # if still NaN after ffill, skip
                    continue

                impact_cost = shares * price * market_impact_factor
                if order_type == 'BUY':
                    total_cost = shares * price + commission + impact_cost
                    if cash >= total_cost:
                        cash -= total_cost
                        holdings[sym] += shares
                elif order_type == 'SELL':
                    if holdings[sym] >= shares:
                        proceeds = shares * price - commission - impact_cost
                        cash += proceeds
                        holdings[sym] -= shares

        # End-of-day valuation
        total_value = cash
        for sym, qty in holdings.items():
            if qty == 0 or sym not in symbol_to_prices:
                continue
            price = symbol_to_prices[sym].loc[current_date]
            if not np.isnan(price):
                total_value += qty * price
        portvals.append(total_value)

    return pd.Series(portvals, index=trading_dates)


def _compute_stats(portvals: pd.Series, benchmark: pd.Series | None = None):
    dr = portvals.pct_change().dropna()
    cum_ret = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    avg_daily_ret = dr.mean() if len(dr) else 0.0
    std_daily_ret = dr.std() if len(dr) else 0.0
    sharpe = (avg_daily_ret / std_daily_ret * np.sqrt(252)) if std_daily_ret > 0 else 0.0

    stats = {
        'cum_ret': float(cum_ret),
        'adr': float(avg_daily_ret),
        'sddr': float(std_daily_ret),
        'sr': float(sharpe),
        'final_val': float(portvals.iloc[-1]),
        'start_date': portvals.index[0],
        'end_date': portvals.index[-1],
        'num_days': len(portvals),
    }

    if benchmark is not None and len(benchmark) > 1:
        # align
        common = portvals.index.intersection(benchmark.index)
        if len(common) > 1:
            b = benchmark.loc[common]
            br = b.pct_change().dropna()
            stats.update({
                'bench_cum_ret': float((b.iloc[-1] / b.iloc[0]) - 1),
                'bench_adr': float(br.mean()) if len(br) else 0.0,
                'bench_sddr': float(br.std()) if len(br) else 0.0,
                'bench_sr': float(((br.mean() / br.std()) * np.sqrt(252)) if br.std() > 0 else 0.0),
            })
    return stats


def test_code(args=None):
    if args is None:
        args = argparse.Namespace(
            orderfile='./orders/orders2.csv',
            start_value=1000000,
            transaction_fee=9.95,
            market_impact_factor=0.005,
            data_dir='data-2011',
            do_verbose=True,
            test_student=0,
            do_plot=True
        )

    print("Running Market Simulator...\n")
    print(f"Orders file: {args.orderfile}")

    # Compute portfolio values
    portvals = compute_portvals(orders_file=args.orderfile, start_val=args.start_value, args=args)
    if portvals.empty:
        print("No portfolio values computed (check data/orders)")
        return

    # Load benchmark (SPY)
    try:
        spy = _load_price_series('SPY', args.data_dir)
        spy = spy.loc[(spy.index >= portvals.index[0]) & (spy.index <= portvals.index[-1])]
    except Exception:
        spy = None

    stats = _compute_stats(portvals, spy)

    print()
    print("--- begin statistics ------------------------------------------------- ")
    print(f"Date Range: {stats['start_date'].date()} to {stats['end_date'].date()} (portfolio)")
    print(f"Number of Trading Days:             {stats['num_days']:14d}")
    print()
    print(f"Cumulative Return of Fund:          {stats['cum_ret']:+.11f}")
    print(f"Average Daily Return of Fund:       {stats['adr']:+.11f}")
    print(f"Volatility (Std Dev):               {stats['sddr']:+.11f}")
    print(f"Sharpe Ratio of Fund:               {stats['sr']:+.11f}")
    print(f"\nFinal Portfolio Value:               {stats['final_val']:+,.2f}")

    if 'bench_cum_ret' in stats:
        theSPY = 'SPY'
        print()
        print(f"Cumulative Return of {theSPY}:           {stats['bench_cum_ret']:+.11f}")
        print(f"Average Daily Return of {theSPY}:        {stats['bench_adr']:+.11f}")
        print(f"Volatility (Std Dev) of {theSPY}:        {stats['bench_sddr']:+.11f}")
        print(f"Sharpe Ratio of {theSPY}:                {stats['bench_sr']:+.11f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Simulator")
    parser.add_argument('-f', '--file', type=str, dest="orderfile", default='./orders/orders-short.csv',
                        help='Path to the order file')
    parser.add_argument('-c', '--cash', type=float, dest="start_value", default=1000000, help='Starting cash')
    parser.add_argument('-t', '--transaction_fee', type=float, dest="transaction_fee", default=9.95,
                        help='Transaction fee')
    parser.add_argument('-m', '--market_impact', type=float, dest="market_impact_factor", default=0.005,
                        help='Market impact factor')
    parser.add_argument('-d', '--data-dir', type=str, dest="data_dir", default='data-2011',
                        help='Directory containing price CSV files')
    parser.add_argument('-v', '--verbose', action='store_true', dest="do_verbose", help='Verbose mode')
    args = parser.parse_args()

    test_code(args)