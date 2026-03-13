# Import
import utils
import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns
from pypfopt.risk_models import CovarianceShrinkage

# -----------------------------
# Get Min Weight
# -----------------------------
def get_min_vol_weights(hist_prices):

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

    ef.min_volatility()

    return pd.Series(ef.clean_weights())

# -----------------------------
# Rolling MinVol Strat
# -----------------------------
def rolling_min_vol(m_prices, m_ret):

    weights_log = {}

    for dt in m_ret.index:

        win_start = dt - pd.DateOffset(years=utils.lookback_years)

        hist = m_prices.loc[win_start:dt]

        if len(hist) < utils.lookback_years * 12:
            continue

        w = get_min_vol_weights(hist)

        weights_log[dt] = w.reindex(m_ret.columns).fillna(0)

    return weights_log

# Monthly data
m_prices = utils.get_monthly_prices(utils.prices)
monthly_ret = utils.get_monthly_returns(utils.prices)

# Rolling Min Vol
weights_log_min_vol = rolling_min_vol(
    m_prices,
    monthly_ret
)

# Equal weight returns
equal_weights_dict = dict(zip(utils.assets, utils.weights))

port_ret_equal = (monthly_ret * equal_weights_dict).sum(axis=1)

# Strategy returns
port_ret_min_vol = utils.compute_portfolio_returns(
    monthly_ret,
    weights_log_min_vol
)

# Weight plots
utils.plot_weight_stack(
    weights_log_min_vol,
    "Rolling 3-Year Minimum Volatility Portfolio Weights"
)

utils.plot_weight_lines(
    weights_log_min_vol,
    "Minimum Volatility Weights by Asset"
)

utils.plot_turnover(
    weights_log_min_vol,
    "Minimum Volatility Portfolio Turnover"
)

# Strategy vs Equal Weight
monthly_contrib = 500

perf = utils.plot_strategy_vs_equal(
    port_ret_strategy=port_ret_min_vol,
    port_ret_equal=port_ret_equal,
    weights_log=weights_log_min_vol,
    monthly_contrib=monthly_contrib,
    strategy_name="Minimum Volatility"
)