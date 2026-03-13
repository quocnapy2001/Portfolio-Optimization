from opt_func_0 import *
from base_case_1 import *

### Benchmark 
# Price
benchmark_ticker = "SPY"

benchmark_prices = yf.download(
    benchmark_ticker,
    start=stockStartDate,
    end=today,
    auto_adjust=True
)["Close"]

benchmark_monthly_prices = benchmark_prices.resample("MS").last()
benchmark_monthly_ret = benchmark_monthly_prices.pct_change().iloc[1:]
benchmark_monthly_ret = benchmark_monthly_ret.squeeze()   # ensure Series

### Portfolio Returns
# Unconstrained weights
weights_df_unconstrained = pd.DataFrame(weights_log).T
weights_df_unconstrained = weights_df_unconstrained.reindex(monthly_ret.index, method="ffill")
port_ret_unconstrained = (monthly_ret * weights_df_unconstrained).sum(axis=1)


# Constrained weights
weights_df_constrained = pd.DataFrame(weights_log_constrained).T
weights_df_constrained = weights_df_constrained.reindex(monthly_ret.index, method="ffill")
port_ret_constrained = (monthly_ret * weights_df_constrained).sum(axis=1)

### Investment Value
value_equal_full = simulate(monthly_contrib, port_ret_equal)
value_unconstrained_full = simulate(monthly_contrib, port_ret_unconstrained)
value_constrained_full = simulate(monthly_contrib, port_ret_constrained)
value_benchmark_full = simulate(monthly_contrib, benchmark_monthly_ret)

### Date Alignment
start_date = min(weights_log.keys())

value_equal = value_equal_full.loc[start_date:].copy()
value_unconstrained = value_unconstrained_full.loc[start_date:].copy()
value_constrained = value_constrained_full.loc[start_date:].copy()
value_benchmark = value_benchmark_full.loc[start_invest_date:].copy()


value_equal.iloc[0] = 0
value_unconstrained.iloc[0] = 0
value_constrained.iloc[0] = 0
value_benchmark.iloc[0] = 0.0

#### Plot 
perf_compare = pd.DataFrame({
    "Equal Weight": value_equal,
    "Unconstrained Max-Sharpe": value_unconstrained,
    "Constrained Max-Sharpe": value_constrained,
    "SPY Passive Benchmark": value_benchmark
})

plt.figure(figsize=(11,6))
perf_compare.plot(ax=plt.gca())

plt.title("Strategy Comparison")
plt.ylabel("Portfolio Value (£)")
plt.xlabel("Date")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

### Plot (Normalised)
norm_compare = perf_compare.copy()
norm_compare = norm_compare.div(norm_compare.iloc[1]).mul(100)

plt.figure(figsize=(11,6))
norm_compare.plot(ax=plt.gca())

plt.title("Normalised Portfolio Performance (Start = 100)")
plt.ylabel("Index Value")
plt.xlabel("Date")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Final Porfolio Value:
print(perf_compare.iloc[-1])