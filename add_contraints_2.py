# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 20:17:10 2026

@author: Owner
"""
from opt_func_0 import *

# ---------------------------------------------------------------
# Rolling 3-yr CONSTRAINED Max-Sharpe (parallel scenario)
# ---------------------------------------------------------------
weights_log_constrained = {}

for dt in m_ret.index:
    win_start = dt - pd.DateOffset(years=lookback_years)
    hist = m_prices.loc[win_start:dt]

    if len(hist) < lookback_years * 12:
        continue

    w = get_max_sharpe_weights(hist, constrained=True)
    weights_log_constrained[dt] = w.reindex(m_ret.columns).fillna(0)

# ---------------------------------------------------------------
# Stacked area chart
# ---------------------------------------------------------------
weights_table_c = pd.DataFrame(weights_log_constrained).T      # rows = month-ends
weights_table_c.index.name = "Month-End"

# Ensure dates are sorted
weights_table_c = weights_table_c.sort_index()

plt.figure(figsize=(12, 6))
weights_table_c.plot(
    kind="area",
    stacked=True,
    figsize=(12, 6),
    alpha=0.9
)

plt.title("Rolling 3-Year Max-Sharpe Portfolio Weights (first live = Jan-2024)")
plt.xlabel("Month-End")
plt.ylabel("Portfolio Weight")
plt.legend(loc="upper left", ncol=2)
plt.grid(True, axis="y")
plt.show()

# Line plot per asset
plt.figure(figsize=(12, 6))

for col in weights_table_c.columns:
    plt.plot(weights_table_c.index, weights_table_c[col], label=col)

plt.title("Rolling 3-Year Max-Sharpe Weights by Asset")
plt.xlabel("Month-End")
plt.ylabel("Weight")
plt.legend(ncol=2)
plt.grid(True)
plt.show()

# Turnover-style plot (how much weights change)
weight_changes_c = weights_table_c.diff().abs().sum(axis=1)

plt.figure(figsize=(10, 4))
plt.plot(weight_changes_c.index, weight_changes_c, color="black")
plt.title("Monthly Portfolio Turnover (L1 Weight Change)")
plt.xlabel("Month-End")
plt.ylabel("Turnover")
plt.grid(True)
plt.show()


# --- 3. Compute *monthly* portfolio returns ----------------------------------
monthly_prices = prices.resample("MS").last()
monthly_ret    = monthly_prices.pct_change().iloc[1:]   # skip NaN first row

equal_weights_dict = dict(zip(assets, weights))
port_ret_equal      = (monthly_ret * equal_weights_dict).sum(axis=1)

weights_df = pd.DataFrame(weights_log_constrained).T
weights_df = weights_df.reindex(monthly_ret.index, method="ffill")
port_ret_max_sharpe = (monthly_ret * weights_df).sum(axis=1)

# --- 4. Simulate £500 contribution at the *start* of every month -------------
def simulate(contributions, port_returns):
    """contributions (£) and port_returns (Series) must align by date index"""
    value = 0.0
    history = []
    for date, r in port_returns.items():
        value = (value + contributions) * (1 + r)   # add £500, then grow
        history.append((date, value))
    return pd.Series(dict(history))

monthly_contrib = 500
value_equal_full = simulate(monthly_contrib, port_ret_equal)
value_max_sharpe_full = simulate(monthly_contrib, port_ret_max_sharpe)

# ------------------------------------------------------------
# Align investment start date (FIRST OPTIMISABLE MONTH)
# ------------------------------------------------------------
start_invest_date = min(weights_log_constrained.keys())

value_equal = value_equal_full.loc[start_invest_date:].copy()
value_max_sharpe = value_max_sharpe_full.loc[start_invest_date:].copy()

# Reset starting capital to zero
value_equal.iloc[0] = 0.0
value_max_sharpe.iloc[0] = 0.0

# Combine for plotting
perf = pd.DataFrame({
    "Equal-Weight": value_equal,
    "Contrained Max-Sharpe":   value_max_sharpe
})

# --- Plot -----------------------------------------------------------------
plt.figure(figsize=(10,6))
perf.plot(ax=plt.gca())
plt.title("£500 Monthly Contribution: Equal-Weight vs. Max-Sharpe Portfolio", fontsize=14)
plt.ylabel("Portfolio Value (£)")
plt.xlabel("Date")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()