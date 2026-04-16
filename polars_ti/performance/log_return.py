# -*- coding: utf-8 -*-
# =============================================================================
# Polars LOG_RETURN Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_log_return(
    close: IntoExpr,
    length: int = 1,
    cumulative: bool = False,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Log Return

    Calculates the logarithmic return of a Series.

    Sources:
        https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period for return calculation. Default: 1
        cumulative: If True, returns cumulative log returns. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Log return expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if cumulative:
        # Cumulative log return: log(close / close[0])
        result = (close_expr / close_expr.first()).log()
        name = f"CUMLOGRET_{length}"
    else:
        # Log return: log(close / close.shift(length))
        result = (close_expr / close_expr.shift(length)).log()
        name = f"LOGRET_{length}"

    # Apply offset if needed
    if offset != 0:
        result = result.shift(offset)

    return result.alias(name)

