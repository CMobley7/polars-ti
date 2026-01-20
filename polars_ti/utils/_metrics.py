# -*- coding: utf-8 -*-
from numpy import log, nan, sqrt
from pandas import Series, Timedelta

from polars_ti.maps import RATE
from polars_ti.utils._math import linear_regression
from polars_ti.utils._time import total_time
from polars_ti.utils._validate import v_series


def cagr(close: Series) -> int | float:
    """Compounded Annual Growth Rate

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ti.cagr(df.close)
    """
    close = v_series(close)
    start, end = close.iloc[0], close.iloc[-1]
    return ((end / start) ** (1 / total_time(close))) - 1


def calmar_ratio(close: Series, method: str = "percent", years: int = 3) -> int | float:
    """The Calmar Ratio is the percent Max Drawdown Ratio 'typically' over
    the past three years.

    Args:
        close (pd.Series): Series of 'close's
        method (str): Max DD calculation options: 'dollar', 'percent', 'log'.
            Default: 'dollar'
        years (int): The positive number of years to use. Default: 3

    >>> result = ti.calmar_ratio(close, method="percent", years=3)
    """
    if years <= 0:
        print(f"[!] calmar_ratio 'years' argument must be greater than zero.")
        return
    close = v_series(close)

    n_years_ago = close.index[-1] - Timedelta(days=365.25 * years)
    close = close[close.index > n_years_ago]

    return cagr(close) / max_drawdown(close, method=method)


def downside_deviation(
    returns: Series, benchmark_rate: int | float = 0.0, tf: str = "years"
) -> int | float:
    """Downside Deviation for the Sortino ratio.
    Benchmark rate is assumed to be annualized. Adjusted according for the
    number of periods per year seen in the data.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        tf (str): Time Frame options: 'days', 'weeks', 'months', and 'years'.
            Default: 'years'

    >>> result = ti.downside_deviation(returns, benchmark_rate=0.0, tf="years")
    """
    # For both de-annualizing the benchmark rate and annualizing result
    returns = v_series(returns)
    days_per_year = returns.shape[0] / total_time(returns, tf)

    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1 / days_per_year)) - 1

    downside = adjusted_benchmark_rate - returns
    downside_sum_of_squares = (downside[downside > 0] ** 2).sum()
    downside_deviation = sqrt(downside_sum_of_squares / (returns.shape[0] - 1))
    return downside_deviation * sqrt(days_per_year)


def jensens_alpha(returns: Series, benchmark_returns: Series) -> int | float:
    """Jensen's 'Alpha' of a series and a benchmark.

    Args:
        returns (pd.Series): Series of 'returns's
        benchmark_returns (pd.Series): Series of 'benchmark_returns's

    >>> result = ti.jensens_alpha(returns, benchmark_returns)
    """
    returns = v_series(returns)
    benchmark_returns = v_series(benchmark_returns)

    benchmark_returns = benchmark_returns.interpolate()
    return linear_regression(benchmark_returns, returns)["a"]


def log_max_drawdown(close: Series) -> int | float:
    """Log Max Drawdown of a series.

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ti.log_max_drawdown(close)
    """
    close = v_series(close)
    log_return = log(close.iloc[-1]) - log(close.iloc[0])
    return log_return - max_drawdown(close, method="log")


def max_drawdown(
    close: Series, method: str | None = None, all: bool = False
) -> int | float:
    """Maximum Drawdown from close. Default: 'dollar'.

    Args:
        close (pd.Series): Series of 'close's
        method (str): Max DD calculation options: 'dollar', 'percent', 'log'.
            Default: 'dollar'
        all (bool): If True, it returns all three methods as a dict.
            Default: False

    >>> result = ti.max_drawdown(close, method="dollar", all=False)
    """
    from polars_ti.performance import drawdown

    close = v_series(close)
    max_dd = drawdown(close).max()

    max_dd_ = {
        "dollar": max_dd.iloc[0],
        "percent": max_dd.iloc[1],
        "log": max_dd.iloc[2],
    }
    if all:
        return max_dd_

    if isinstance(method, str) and method in max_dd_.keys():
        return max_dd_[method]
    return max_dd_["dollar"]


def optimal_leverage(
    close: Series,
    benchmark_rate: int | float = 0.0,
    period: int | float = RATE["TRADING_DAYS_PER_YEAR"],
    log: bool = False,
    capital: int | float = 1.0,
    **kwargs: dict,
) -> int | float:
    """Optimal Leverage of a series. NOTE: Incomplete. Do NOT use.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        period (int, float): Period to use to calculate Mean Annual Return
            and Annual Standard Deviation.
            Default: None or the default sharpe_ratio.period()
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False

    >>> result = ti.optimal_leverage(close, benchmark_rate=0.0, log=False)
    """
    from polars_ti.performance import log_return, percent_return

    close = v_series(close)

    use_cagr = kwargs.pop("use_cagr", False)
    if log:
        returns = log_return(close=close)
    else:
        returns = percent_return(close=close)
    # sharpe = sharpe_ratio(close, benchmark_rate=benchmark_rate, log=log, use_cagr=use_cagr, period=period)

    period_mu = period * returns.mean()
    period_std = sqrt(period) * returns.std()

    mean_excess_return = period_mu - benchmark_rate
    # sharpe = mean_excess_return / period_std
    opt_leverage = (period_std**-2) * mean_excess_return

    amount = int(capital * opt_leverage)
    return amount


def pure_profit_score(close: Series) -> int | float:
    """Pure Profit Score of a series.

    Args:
        close (pd.Series): Series of 'close's

    >>> result = ti.pure_profit_score(df.close)
    """
    close = v_series(close)
    close_index = Series(0, index=close.reset_index().index)

    r = linear_regression(close_index, close)["r"]
    if r is not nan:
        return r * cagr(close)
    return 0.0


def sharpe_ratio(
    close: Series,
    benchmark_rate: int | float = 0.0,
    log: bool = False,
    use_cagr: bool = False,
    period: int | float = RATE["TRADING_DAYS_PER_YEAR"],
) -> int | float:
    """Sharpe Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False
        use_cagr (bool): Use cagr - benchmark_rate instead. Default: False
        period (int, float): Period to use to calculate Mean Annual Return and
            Annual Standard Deviation.
            Default: RATE["TRADING_DAYS_PER_YEAR"] (currently 252)

    >>> result = ti.sharpe_ratio(close, benchmark_rate=0.0, log=False)
    """
    from polars_ti.performance import log_return, percent_return

    close = v_series(close)
    if log:
        returns = log_return(close=close)
    else:
        returns = percent_return(close=close)

    if use_cagr:
        return cagr(close) / volatility(close, returns, log=log)
    else:
        period_mu = period * returns.mean()
        period_std = sqrt(period) * returns.std()
        return (period_mu - benchmark_rate) / period_std


def sortino_ratio(
    close: Series,
    benchmark_rate: int | float = 0.0,
    log: bool = False,
) -> int | float:
    """Sortino Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False

    >>> result = ti.sortino_ratio(close, benchmark_rate=0.0, log=False)
    """
    from polars_ti.performance import log_return, percent_return

    close = v_series(close)

    if log:
        returns = log_return(close=close)
    else:
        returns = percent_return(close=close)

    result = (cagr(close) - benchmark_rate) / downside_deviation(returns)
    return result


def volatility(
    close: Series,
    tf: str = "years",
    returns: bool = False,
    log: bool = False,
    **kwargs: dict,
) -> int | float:
    """Volatility of a series. Default: 'years'

    Args:
        close (pd.Series): Series of 'close's
        tf (str): Time Frame options: 'days', 'weeks', 'months', and 'years'.
            Default: 'years'
        returns (bool): If True, then it replace the close Series with the
            user defined Series; typically user generated returns or percent
            returns or log returns. Default: False
        log (bool): If True, calculates log_return. Otherwise it calculates
            percent_return. Default: False

    >>> result = ti.volatility(close, tf="years", returns=False, log=False)
    """
    from polars_ti.performance import log_return, percent_return

    close = v_series(close)

    if not returns:
        if log:
            returns = log_return(close=close)
        else:
            returns = percent_return(close=close)
    else:
        returns = close

    factor = returns.shape[0] / total_time(returns, tf)
    if kwargs.pop("nearest_day", False) and tf.lower() == "years":
        factor = int(factor + 1)

    return sqrt(factor) * returns.std()
