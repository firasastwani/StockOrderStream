import pandas as pd
from datetime import datetime
import argparse
import importlib
import sys
import re

sys.path.insert(0, './portfolio')
from portfolio.get_data import get_data, plot_data
from portfolio.assess_portfolio import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, args=None):
    """
    Placeholder for compute_portvals. This function should compute the daily portfolio value
    given a sequence of orders in a CSV file. Currently, it just returns a placeholder value.

    Parameters
    ----------
    orders_file : str
        CSV file to read orders from.
    start_val : float
        Starting cash value.

    Returns
    -------
    portvals : pd.Series
        Placeholder for daily portfolio value for each trading day.
    """

    # Placeholder for commission and market impact values from args
    commission = args.transaction_fee if args else 9.95
    market_impact_factor = args.market_impact_factor if args else 0.005
    verbose = args.do_verbose if args else False

    # Placeholder for actual portfolio values
    dates = pd.date_range(start="2021-01-01", end="2021-12-31")
    portvals = pd.Series([start_val] * len(dates), index=dates)  # Placeholder portfolio values

    if verbose:
        print("\nFinal Portfolio Value Per Day (Stub Output):\n")
        print(portvals.tail())

    return portvals


def test_code(args=None):
    if args is None:
        args = argparse.Namespace(
            orderfile='./orders/orders2.csv',
            start_value=1000000,
            transaction_fee=9.95,
            market_impact_factor=0.005,
            do_verbose=True,
            test_student=0,
            do_plot=True
        )

    # Print initial message
    print("Running Market Simulator (Stub)...\n")
    print(f"Orders file: {args.orderfile}")

    # Compute portfolio values (Stub)
    portvals = compute_portvals(orders_file=args.orderfile, start_val=args.start_value, args=args)

    # Get a placeholder date range for portfolio analysis
    start_date = portvals.index.min()
    end_date = portvals.index.max()

    # Stub for the SPY values and comparison metrics (Sharpe Ratio, etc.)
    sharpe_ratio = 0.5  # Placeholder value
    cum_ret = 0.05  # Placeholder value
    avg_daily_ret = 0.0005  # Placeholder value
    std_daily_ret = 0.002  # Placeholder value
    sharpe_ratio_SPY = 0.6  # Placeholder value
    cum_ret_SPY = 0.06  # Placeholder value
    avg_daily_ret_SPY = 0.0006  # Placeholder value
    std_daily_ret_SPY = 0.0021  # Placeholder value

    # use $SPX or SPY
    theSPY = '$SPX'
    # Print Portfolio vs SPY statistics (Stub)
    print()
    print("--- begin statistics ------------------------------------------------- ")
    print(f"Date Range: {start_date.date()} to {end_date.date()} (portfolio)")
    print(f"Number of Trading Days:             {len(portvals):14d}")

    print()
    print(f"Cumulative Return of Fund:          {cum_ret:+.11f}")
    print(f"Average Daily Return of Fund:       {avg_daily_ret:+.11f}")
    print(f"Volatility (Std Dev):               {std_daily_ret:+.11f}")
    print(f"Sharpe Ratio of Fund:               {sharpe_ratio:+.11f}")
    final_portval = portvals.iloc[-1]
    print(f"\nFinal Portfolio Value (Stub):        {final_portval:+,.2f}")

    print()
    print(f"Cumulative Return of {theSPY}:           {cum_ret_SPY:+.11f}")
    print(f"Average Daily Return of {theSPY}:        {avg_daily_ret_SPY:+.11f}")
    print(f"Volatility (Std Dev) of {theSPY}:        {std_daily_ret_SPY:+.11f}")
    print(f"Sharpe Ratio of {theSPY}:                {sharpe_ratio_SPY:+.11f}")

    if args.do_plot:
        # Stub for plot logic
        print("\nPlotting is enabled (Stub)...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Simulator (Stub)")
    parser.add_argument('-f', '--file', type=str, dest="orderfile", default='./orders/orders-short.csv',
                        help='Path to the order file')
    parser.add_argument('-c', '--cash', type=float, dest="start_value", default=1000000, help='Starting cash')
    parser.add_argument('-t', '--transaction_fee', type=float, dest="transaction_fee", default=9.95,
                        help='Transaction fee')
    parser.add_argument('-m', '--market_impact', type=float, dest="market_impact_factor", default=0.005,
                        help='Market impact factor')
    parser.add_argument('-v', '--verbose', action='store_true', dest="do_verbose", help='Verbose mode')
    parser.add_argument('-p', '--plot', action='store_true', dest="do_plot", help='Plot results')
    args = parser.parse_args()

    test_code(args)