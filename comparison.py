# max_sharpe_strategy_comparison.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

import utils
import ms_unconstrained
import ms_constrained
import minVol


# -----------------------------
# Benchmark
# -----------------------------

benchmark_prices = yf.download(
    "SPY",
    start=utils.stockStartDate,
    end=utils.today,
    auto_adjust=True
)["Close"]

benchmark_ret = utils.get_monthly_returns(benchmark_prices).squeeze()

# -----------------------------
# Simulate Portfolios
# -----------------------------

monthly_contrib = ms_unconstrained.monthly_contrib

value_equal_full = utils.simulate(
    monthly_contrib,
    ms_unconstrained.port_ret_equal
)

value_unconstrained_full = utils.simulate(
    monthly_contrib,
    ms_unconstrained.port_ret_unconstrained
)

value_constrained_full = utils.simulate(
    monthly_contrib,
    ms_constrained.port_ret_constrained
)

value_benchmark_full = utils.simulate(
    monthly_contrib,
    benchmark_ret
)

value_min_vol_full = utils.simulate(
    monthly_contrib,
    minVol.port_ret_min_vol
)

# -----------------------------
# Align Start Date
# -----------------------------

start_date = min(ms_unconstrained.weights_log_unconstrained.keys())

value_equal = value_equal_full.loc[start_date:].copy()
value_unconstrained = value_unconstrained_full.loc[start_date:].copy()
value_constrained = value_constrained_full.loc[start_date:].copy()
value_benchmark = value_benchmark_full.loc[start_date:].copy()
value_min_vol = value_min_vol_full.loc[start_date:].copy()

value_equal.iloc[0] = 0
value_unconstrained.iloc[0] = 0
value_constrained.iloc[0] = 0
value_benchmark.iloc[0] = 0
value_min_vol.iloc[0] = 0


# -----------------------------
# Combine Strategies
# -----------------------------

perf_compare = pd.DataFrame({

    "Equal Weight": value_equal,
    "Unconstrained Max-Sharpe": value_unconstrained,
    "Constrained Max-Sharpe": value_constrained,
    "Minimum Volatility": value_min_vol,
    "SPY Benchmark": value_benchmark

})


# -----------------------------
# Plot Performance
# -----------------------------

plt.figure(figsize=(11,6))

perf_compare.plot(ax=plt.gca())

plt.title("Strategy Comparison")
plt.ylabel("Portfolio Value (£)")
plt.xlabel("Date")

plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# -----------------------------
# Normalised Performance
# -----------------------------

norm_compare = perf_compare.div(perf_compare.iloc[1]).mul(100)

plt.figure(figsize=(11,6))

norm_compare.plot(ax=plt.gca())

plt.title("Normalised Portfolio Performance (Start = 100)")
plt.ylabel("Index Value")
plt.xlabel("Date")

plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# -----------------------------
# Final Portfolio Values
# -----------------------------

print("\nFinal Portfolio Value:")
print(perf_compare.iloc[-1])