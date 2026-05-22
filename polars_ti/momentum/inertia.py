# -*- coding: utf-8 -*-
# =============================================================================
# Polars Inertia Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_inertia(
    close: IntoExpr,
    high: IntoExpr | None = None,
    low: IntoExpr | None = None,
    length: int = 20,
    rvi_length: int = 14,
    scalar: float = 100.0,
    refined: bool = False,
    thirds: bool = False,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: Inertia (INERTIA)

    RVI smoothed by Least Squares Moving Average.
    Positive Inertia > 50, Negative Inertia < 50.

    Args:
        close: Column name or pl.Expr for 'close' prices
        high: Optional for refined/thirds mode
        low: Optional for refined/thirds mode
        length: LSMA period. Default: 20
        rvi_length: RVI period. Default: 14
        scalar: RVI scalar. Default: 100
        refined: Use refined RVI. Default: False
        thirds: Use thirds RVI. Default: False
        mamode: MA mode for RVI. Default: 'ema'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Inertia expression
    """
    from polars_ti.volatility.rvi import pl_rvi
    from polars_ti.overlap.linreg import pl_linreg
    import numpy as np

    close_expr = v_expr(close)
    _length = length
    _rvi_length = rvi_length
    _scalar = scalar
    _refined = refined
    _thirds = thirds
    _mamode = mamode

    # For simple (non-refined, non-thirds) case only close is needed
    if refined or thirds:
        high_expr = v_expr(high)
        low_expr = v_expr(low)

        def compute(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            c = df["close"]
            h = df["high"]
            l = df["low"]

            # Compute RVI
            if _refined:
                rvi_arr = df.select(
                    pl_rvi(
                        "close",
                        high="high",
                        low="low",
                        length=_rvi_length,
                        scalar=_scalar,
                        refined=True,
                        mamode=_mamode,
                    )
                )[df.columns[0]]
            else:  # thirds
                rvi_arr = df.select(
                    pl_rvi(
                        "close",
                        high="high",
                        low="low",
                        length=_rvi_length,
                        scalar=_scalar,
                        thirds=True,
                        mamode=_mamode,
                    )
                )[df.columns[0]]

            # Apply linreg
            rvi_df = pl.DataFrame({"rvi": rvi_arr})
            result = rvi_df.select(pl_linreg("rvi", length=_length))
            return result.to_series()

        struct_expr = pl.struct(close=close_expr, high=high_expr, low=low_expr)
        inertia_expr = struct_expr.map_batches(compute, return_dtype=pl.Float64)
        _mode = "r" if refined else "t"
    else:
        # Simple case - compute RVI then apply linreg in map_batches
        def compute_simple(s: pl.Series) -> pl.Series:
            df = pl.DataFrame({"close": s})
            rvi_result = df.select(pl_rvi("close", length=_rvi_length, scalar=_scalar, mamode=_mamode))
            rvi_col = rvi_result.to_series()
            rvi_df = pl.DataFrame({"rvi": rvi_col})
            linreg_result = rvi_df.select(pl_linreg("rvi", length=_length))
            return linreg_result.to_series()

        inertia_expr = close_expr.map_batches(compute_simple, return_dtype=pl.Float64)
        _mode = ""

    if offset != 0:
        inertia_expr = inertia_expr.shift(offset)

    _props = f"_{length}_{rvi_length}"
    return inertia_expr.alias(f"INERTIA{_mode}{_props}")
