# -*- coding: utf-8 -*-
# =============================================================================
# Polars PERCENT_RETURN Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def percent_return(
    close: IntoExpr,
    length: int = 1,
    cumulative: bool = False,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Percent Return

    Calculates the percent return of a Series.

    Sources:
        https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period for return calculation. Default: 1
        cumulative: If True, returns cumulative percent returns. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Percent return expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if cumulative:
        # Cumulative percent return: (close / close[0]) - 1
        result = (close_expr / close_expr.first()) - 1
        name = f"CUMPCTRET_{length}"
    else:
        # Percent return: (close / close.shift(length)) - 1
        result = (close_expr / close_expr.shift(length)) - 1
        name = f"PCTRET_{length}"

    # Apply offset if needed
    if offset != 0:
        result = result.shift(offset)

    return result.alias(name)
