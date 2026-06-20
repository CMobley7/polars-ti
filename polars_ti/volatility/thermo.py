# -*- coding: utf-8 -*-
# =============================================================================
# Polars THERMO Implementation (Pure Composition)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def thermo(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 20,
    long: float = 2.0,
    short: float = 0.5,
    mamode: str = "ema",
    asint: bool = True,
    drift: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Elders Thermometer (THERMO)

    Pure composition using pl_ma. Measures price volatility.

    Sources:
        https://www.motivewave.com/studies/elders_thermometer.htm
        https://www.tradingview.com/script/HqvTuEMW-Elder-s-Market-Thermometer-LazyBear/

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        length: The period. Default: 20
        long: The buy factor. Default: 2.0
        short: The sell factor. Default: 0.5
        mamode: MA type. Default: 'ema'
        asint: Return signals as int. Default: True
        drift: The diff period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with thermo, thermo_ma, thermo_long, thermo_short columns
    """
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)

    if high_expr is None or low_expr is None:
        return None

    # Calculate thermoL and thermoH
    thermo_l = (low_expr.shift(drift) - low_expr).abs()
    thermo_h = (high_expr - high_expr.shift(drift)).abs()

    # thermo = max(thermoL, thermoH) - using when/then/otherwise
    thermo = pl.when(thermo_h < thermo_l).then(thermo_l).otherwise(thermo_h)

    # MA of thermo
    thermo_ma = ma(name=mamode, source=thermo, length=length, talib=False)

    # Long/Short signals
    thermo_long_cond = thermo < (thermo_ma * pl.lit(long))
    thermo_short_cond = thermo > (thermo_ma * pl.lit(short))

    # Cast to int if requested
    if asint:
        thermo_long_val = thermo_long_cond.cast(pl.Int64)
        thermo_short_val = thermo_short_cond.cast(pl.Int64)
    else:
        thermo_long_val = thermo_long_cond
        thermo_short_val = thermo_short_cond

    # Apply offset
    if offset != 0:
        thermo = thermo.shift(offset)
        thermo_ma = thermo_ma.shift(offset)
        thermo_long_val = thermo_long_val.shift(offset)
        thermo_short_val = thermo_short_val.shift(offset)

    _props = f"_{length}_{int(long)}_{short}"

    return pl.struct(
        [
            thermo.alias("thermo"),
            thermo_ma.alias("thermo_ma"),
            thermo_long_val.alias("thermo_long"),
            thermo_short_val.alias("thermo_short"),
        ]
    ).alias(f"THERMO{_props}")
