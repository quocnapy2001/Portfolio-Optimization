import utils
import pandas as pd
import matplotlib.pyplot as plt


# Monthly data
m_prices = utils.get_monthly_prices(utils.prices)
monthly_ret = utils.get_monthly_returns(utils.prices)


# Rolling Max-Sharpe
weights_log_unconstrained = utils.rolling_max_sharpe(
    m_prices,
    monthly_ret,
    constrained=False
)


# Equal weight returns
equal_weights_dict = dict(zip(utils.assets, utils.weights))

port_ret_equal = (monthly_ret * equal_weights_dict).sum(axis=1)


# Max-Sharpe returns
port_ret_unconstrained = utils.compute_portfolio_returns(
    monthly_ret,
    weights_log_unconstrained
)

# -----------------------------
# Weight Over Time
# -----------------------------
utils.plot_weight_stack(
    weights_log_unconstrained,
    "Rolling 3-Year Unconstrained Max-Sharpe Portfolio Weights"
)

utils.plot_weight_lines(
    weights_log_unconstrained,
    "Unconstrained Max-Sharpe Weights by Asset"
)

utils.plot_turnover(
    weights_log_unconstrained,
    "Unconstrained Portfolio Turnover"
)


# -----------------------------
# Plot
# -----------------------------
monthly_contrib = 500

perf = utils.plot_strategy_vs_equal(
    port_ret_strategy=port_ret_unconstrained,
    port_ret_equal=port_ret_equal,
    weights_log=weights_log_unconstrained,
    monthly_contrib=monthly_contrib,
    strategy_name="Unconstrained Max-Sharpe"
)