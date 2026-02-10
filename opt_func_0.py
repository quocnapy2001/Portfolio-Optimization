# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 18:32:39 2026

@author: Owner
"""

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


# Stock Pick
assets = ["NVDA", "ASTS","PLTR", "KO", "BTI", "FANG", "LYG", "GOOGL"]

# Assign Weights
weights = np.full(len(assets), 1 / len(assets))
print(weights)

# Starting and Ending date
stockStartDate = '2021-01-01'
today = datetime.today().strftime("%Y-%m-%d")

print("Starting Date: ",stockStartDate)
print("Ending Date: ", today)

# DataFrame of adjusted Close Price
# This downloads ALL tickers at once
prices = pd.DataFrame()
prices = yf.download(
    assets, 
    start=stockStartDate, 
    end=today,
    auto_adjust=True # Take into account adjustment such as Dividends
)['Close']

# Visual of stock
title = "Portfolio Close Price History"

my_stocks = prices 

for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label = c)
    
plt.title(title, fontsize = 18)
plt.xlabel("Date", fontsize = 14)
plt.ylabel("Close Price", fontsize = 14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),  # x = 1 means just outside right edge
    title="Tickers"
)

plt.show()

# Daily simple returns:
returns = prices.pct_change()
returns

# Annualized Covariance Matrix:
cov_matrix_annual = returns.cov() * 252
cov_matrix_annual

# Portfolio Variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

# Portfolio Standard Deviation/Risk
port_volatility = np.sqrt(port_variance)

# Annual Portfolio Returns
portSimpleAnnualReturn = np.sum(returns.mean() * weights) *252

# Expected Annual Return, Risk and Variance
percent_var = str(round(port_variance * 100, 1)) + "%"
percent_vols = str(round(port_volatility * 100, 1)) + "%"
percent_ret = str(round(portSimpleAnnualReturn * 100, 1)) + "%"

# Sharpe Ratio
risk_free_rate = 0.0
# ------------------------------------------------------------
# Equal-Weight Metrics (MONTHLY – consistent with rolling model)
# ------------------------------------------------------------
monthly_prices_static = prices.resample("MS").first()
monthly_ret_static = monthly_prices_static.pct_change().dropna()

ann_ret = np.sum(monthly_ret_static.mean() * weights) * 12
ann_vol = np.sqrt(
    np.dot(
        weights.T,
        np.dot(monthly_ret_static.cov() * 12, weights)
    )
)

sharpe_ratio_monthly = ann_ret / ann_vol

print("\nEqual-Weight Metrics (Monthly-based):")
print("Annual Return:", round(ann_ret * 100, 2), "%")
print("Annual Volatility:", round(ann_vol * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio_monthly, 2))

# Static Optimization

# Portfolio Optimization # Expected return and annualised sample covariance matrix 
mu = expected_returns.mean_historical_return(prices, frequency=12) 
S = risk_models.sample_cov(prices, frequency=12) 

# Max Sharpe 
ef = EfficientFrontier(mu, S) 
ef_weights = ef.max_sharpe() 
cleaned_weights = ef.clean_weights()

print("\nBased Max Sharp Metrics (Monthly-based):")
print(cleaned_weights)
# Efficient Performance:
ef.portfolio_performance(verbose=True)

# Sanity Check
w = pd.Series(cleaned_weights)

print("\nSum of weights:", w.sum())
print(w.sort_values(ascending=False))

w = pd.Series(cleaned_weights)

plt.figure(figsize=(8,4))
w.sort_values(ascending=False).plot(kind="bar")
plt.title("Static Max-Sharpe Optimised Weights")
plt.ylabel("Weight")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# What next?
#- Add weight contraints
#- Max-Sharpe vs Min-Vol

### Optimization with Rebalancing

# ---------------------------------------------------------------
# Monthly Optimization Function
# ---------------------------------------------------------------
def get_max_sharpe_weights(hist_prices, constrained=False):
    mu = expected_returns.mean_historical_return(hist_prices, frequency=12)
    S  = risk_models.sample_cov(hist_prices, frequency=12)

    ef = EfficientFrontier(mu, S)

    if constrained:
        ef.add_constraint(lambda w: w >= 0.05)   # min 5%
        ef.add_constraint(lambda w: w <= 0.20)   # max 20%
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
    ef.max_sharpe()
    return pd.Series(ef.clean_weights())

# Look back period
lookback_years  = 3