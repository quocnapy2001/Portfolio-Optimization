# Import
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models 
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.risk_models import CovarianceShrinkage


# Stock Pick
assets = ["NVDA", "ASTS","PLTR", "KO", "BTI", "FANG", "LYG", "GOOGL"]

# Assign Weights
weights = np.full(len(assets), 1 / len(assets))
print(weights)

# Starting and Ending date
stockStartDate = '2021-01-01'
today = '2026-01-01'

print("Starting Date: ",stockStartDate)
print("Ending Date: ", today)

# Look back period
lookback_years  = 3

# -------------------------
# Price data
# -------------------------
prices = yf.download(
    assets,
    start=stockStartDate,
    end = today,
    auto_adjust=True
)["Close"]

# ------------------------
# Price Plot
# ------------------------
plt.figure(figsize=(12,6))

for col in prices.columns:
    plt.plot(prices.index, prices[col], label=col)

plt.title("Asset Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend(ncol=2)

plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# -------------------------
# Helper Functions
# -------------------------

def get_monthly_prices(prices):
    return prices.resample("MS").first()


def get_monthly_returns(prices):
    return prices.resample("MS").first().pct_change().dropna()


def get_max_sharpe_weights(hist_prices, constrained=False):

    mu = expected_returns.ema_historical_return(
        hist_prices,
        frequency=12,
        span=12
    )

    S = CovarianceShrinkage(
        hist_prices,
        frequency=12
    ).ledoit_wolf()

    ef = EfficientFrontier(mu, S)

    if constrained:

        ef.add_constraint(lambda w: w >= 0.05)
        ef.add_constraint(lambda w: w <= 0.20)

        ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    ef.max_sharpe()

    return pd.Series(ef.clean_weights())


def rolling_max_sharpe(m_prices, m_ret, constrained=False):

    weights_log = {}

    for dt in m_ret.index:

        win_start = dt - pd.DateOffset(years=lookback_years)

        hist = m_prices.loc[win_start:dt]

        if len(hist) < lookback_years * 12:
            continue

        w = get_max_sharpe_weights(hist, constrained)

        weights_log[dt] = w.reindex(m_ret.columns).fillna(0)

    return weights_log


def compute_portfolio_returns(monthly_ret, weights_log):

    weights_df = pd.DataFrame(weights_log).T

    weights_df = weights_df.reindex(monthly_ret.index, method="ffill")

    return (monthly_ret * weights_df).sum(axis=1)


def simulate(contribution, port_returns):

    value = 0
    values = []

    for r in port_returns:
        value = (value + contribution) * (1 + r)
        values.append(value)

    # keep original datetime index
    return pd.Series(values, index=port_returns.index)


def plot_strategy_vs_equal(port_ret_strategy,
                           port_ret_equal,
                           weights_log,
                           monthly_contrib,
                           strategy_name):
    
    # Simulate investment
    value_equal_full = simulate(
        monthly_contrib,
        port_ret_equal
    )

    value_strategy_full = simulate(
        monthly_contrib,
        port_ret_strategy
    )

    # Align start date
    start_date = min(weights_log.keys())

    value_equal = value_equal_full.loc[start_date:].copy()
    value_strategy = value_strategy_full.loc[start_date:].copy()

    value_equal.iloc[0] = 0
    value_strategy.iloc[0] = 0

    # Combine results
    perf = pd.DataFrame({
        "Equal Weight": value_equal,
        strategy_name: value_strategy
    })

    # Plot
    plt.figure(figsize=(10,6))

    perf.plot(ax=plt.gca())

    plt.title(f"£500 Monthly Contribution: Equal-Weight vs {strategy_name}")
    plt.ylabel("Portfolio Value (£)")
    plt.xlabel("Date")

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    return perf


def plot_weight_stack(weights_log, title):

    weights_table = pd.DataFrame(weights_log).T
    weights_table.index.name = "Month-End"
    weights_table = weights_table.sort_index()

    plt.figure(figsize=(12,6))

    weights_table.plot(
        kind="area",
        stacked=True,
        ax=plt.gca(),
        alpha=0.9
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")

    plt.legend(loc="upper left", ncol=2)

    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()



def plot_weight_lines(weights_log, title):

    weights_table = pd.DataFrame(weights_log).T
    weights_table = weights_table.sort_index()

    plt.figure(figsize=(12,6))

    for col in weights_table.columns:
        plt.plot(weights_table.index, weights_table[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Weight")

    plt.legend(ncol=2)

    plt.grid(True)

    plt.tight_layout()
    plt.show()



def plot_turnover(weights_log, title):

    weights_table = pd.DataFrame(weights_log).T
    weights_table = weights_table.sort_index()

    turnover = weights_table.diff().abs().sum(axis=1)

    plt.figure(figsize=(10,4))

    plt.plot(turnover.index, turnover, color="black")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Turnover")

    plt.grid(True)

    plt.tight_layout()
    plt.show()