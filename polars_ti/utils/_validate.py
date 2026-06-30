# -*- coding: utf-8 -*-
from functools import partial

from polars_ti._typing import (
    Float,
    Int,
    IntFloat,
    List,
    Optional,
    np_floating,
    np_integer,
)


def is_percent(x: IntFloat) -> bool:
    if isinstance(x, (float, int, np_floating, np_integer)):
        return x is not None and 0 <= x <= 100
    return False


def v_bool(var: bool, default: bool = True) -> bool:
    """Returns default=True if var is not a bool."""
    if isinstance(var, bool):
        return bool(var)
    return default


def v_dataframe(obj) -> None:
    """Legacy validator — no-op retained for API compatibility."""
    pass


def v_float(var: IntFloat, default: IntFloat, ne: Optional[IntFloat] = 0.0) -> Float:
    """Returns the default if var is not equal to the ne value."""
    _types = (float, int, np_floating, np_integer)
    if isinstance(ne, _types) and isinstance(var, _types):
        if float(var) != float(ne):
            return float(var)
    return float(default)


def v_int(var: Int, default: Int, ne: Optional[Int] = 0) -> Int:
    """Returns the default if var is not equal to the ne value."""
    if isinstance(var, int) and int(var) != int(ne):
        return int(var)
    if isinstance(var, np_integer) and var.item() != int(ne):
        return var.item()
    return int(default)


def v_str(var: str, default: str) -> str:
    """ "Returns the default value if var is not a empty str"""
    if isinstance(var, str) and len(var) > 0:
        return f"{var}"
    return f"{default}"


def v_ascending(var: bool) -> bool:
    """Returns True by default"""
    return partial(v_bool, default=True)(var=var)


def v_datetime_ordered(df) -> bool:
    """Legacy validator — always returns False (pandas index ordering not applicable)."""
    return False


def v_drift(var: Int) -> Int:
    """Defaults to 1"""
    return partial(v_int, default=1, ne=0)(var=var)


def v_list(var: List, default: List = []) -> List:
    """Returns [] if not a valid list"""
    if isinstance(var, list) and len(var) > 0:
        return var
    return default


def v_lowerbound(
    var: IntFloat,
    bound: IntFloat = 0,
    default: IntFloat = 0,
    strict: bool = True,
    complement: bool = False,
) -> IntFloat:
    """Returns the default if var(iable) not greater(equal) than bound."""
    var_type = None
    if isinstance(var, (float, np_floating)):
        var_type = float
    if isinstance(var, (int, np_integer)):
        var_type = int

    if var_type is None:
        return default

    valid = False
    if strict:
        valid = var_type(var) > var_type(bound)
    else:
        valid = var_type(var) >= var_type(bound)

    if complement:
        valid = not valid

    if valid:
        return var_type(var)
    return default


def v_mamode(var: str, default: str) -> str:  # Could be an alias.
    return v_str(var, default)


def v_offset(var: Int) -> Int:
    """Defaults to 0"""
    return partial(v_int, default=0, ne=0)(var=var)


def v_pos_default(var: IntFloat, default: IntFloat = 0, strict: bool = True, complement: bool = False) -> IntFloat:
    return partial(v_lowerbound, bound=0)(var=var, default=default, strict=strict, complement=complement)


def v_scalar(var: IntFloat, default: Optional[IntFloat] = 1) -> Float:
    """Returns the default if var is not a float."""
    if isinstance(var, (float, int, np_floating, np_integer)):
        return float(var)
    return float(default)


def v_series(series, length: Optional[IntFloat] = 0):
    """Legacy validator — returns the series unchanged if it has sufficient length.

    Accepts any sequence-like object; size checked via len() for duck-typed compat.
    """
    if series is None:
        return None
    try:
        size = len(series)
    except TypeError:
        return None
    if size >= v_pos_default(length, 0):
        return series
    return None


def v_talib(var: bool) -> bool:
    """Returns True by default"""
    return partial(v_bool, default=True)(var=var)


def v_tradingview(var: bool) -> bool:
    """Returns True by default"""
    return partial(v_bool, default=True)(var=var)


def v_upperbound(var: IntFloat, bound: IntFloat = 0, default: IntFloat = 0, strict: bool = True) -> IntFloat:
    return partial(v_lowerbound, complement=True)(var=var, bound=bound, default=default, strict=strict)


# =============================================================================
# Polars Validators (for Polars-TI conversion)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr, PlExprOpt, PolarsFrame


def v_expr(expr: IntoExpr, length: int = 0) -> PlExprOpt:
    """Validate and convert column name or expression to pl.Expr.

    Args:
        expr: Column name (str) or Polars expression (pl.Expr)
        length: Minimum required length (unused for expressions, kept for API parity)

    Returns:
        pl.Expr if valid, None otherwise
    """
    if isinstance(expr, str):
        return pl.col(expr)
    if isinstance(expr, pl.Expr):
        return expr
    return None


def v_polars_frame(obj) -> None:
    """Validate that obj is a Polars DataFrame or LazyFrame.

    Raises:
        TypeError: If obj is not a Polars DataFrame or LazyFrame
    """
    if not isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"Requires a Polars DataFrame or LazyFrame, got {type(obj).__name__}")


def v_polars_series(series: pl.Series | None, length: int = 0) -> pl.Series | None:
    """Validate Polars Series meets minimum length requirement.

    Args:
        series: Polars Series to validate
        length: Minimum required length

    Returns:
        The series if valid, None otherwise
    """
    if series is not None and isinstance(series, pl.Series):
        if len(series) >= v_pos_default(length, 0):
            return series
    return None
