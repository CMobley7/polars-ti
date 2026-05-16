# -*- coding: utf-8 -*-
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr


def pl_log_return(close: IntoExpr) -> PlExpr:
    """Polars: Calculate log returns as an expression."""
    from polars_ti.utils._validate import v_expr

    close_expr = v_expr(close)
    return close_expr.log().diff().alias("log_return")


def pl_percent_return(close: IntoExpr) -> PlExpr:
    """Polars: Calculate percent returns as an expression."""
    from polars_ti.utils._validate import v_expr

    close_expr = v_expr(close)
    return close_expr.pct_change().alias("pct_return")


def pl_cumulative_return(close: IntoExpr) -> PlExpr:
    """Polars: Calculate cumulative returns as an expression."""
    from polars_ti.utils._validate import v_expr

    close_expr = v_expr(close)
    first = close_expr.first()
    return ((close_expr / first) - 1).alias("cum_return")


def pl_rolling_volatility(close: IntoExpr, length: int = 20) -> PlExpr:
    """Polars: Rolling volatility (standard deviation of returns) as expression."""
    from polars_ti.utils._validate import v_expr

    close_expr = v_expr(close)
    returns = close_expr.pct_change()
    return returns.rolling_std(window_size=length).alias(f"volatility_{length}")


def pl_drawdown(close: IntoExpr) -> PlExpr:
    """Polars: Calculate drawdown from peak as an expression."""
    from polars_ti.utils._validate import v_expr

    close_expr = v_expr(close)
    peak = close_expr.cum_max()
    return ((close_expr - peak) / peak).alias("drawdown")


def pl_max_drawdown(df: pl.DataFrame, close_col: str) -> float:
    """Polars: Calculate maximum drawdown value."""
    close = df[close_col]
    peak = close.cum_max()
    drawdown = (close - peak) / peak
    return drawdown.min()
