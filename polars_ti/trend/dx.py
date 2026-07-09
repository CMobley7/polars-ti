# -*- coding: utf-8 -*-
# =============================================================================
# Polars DX (Directional Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def dx(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    mamode: str = "rma",
    talib: bool = True,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Directional Index (DX)

    The Directional Index (DX) is an intermediate step in calculating the
    Average Directional Index (ADX). It measures the strength of trend
    direction by comparing positive and negative directional movements.

    DX = scalar * |DMP - DMN| / (DMP + DMN)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        scalar: Magnification factor. Default: 100
        mamode: Smoothing MA type. Default: 'rma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        drift: Difference period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: DX expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib, v_mamode
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _mamode = v_mamode(mamode, "rma")
    # TA-Lib's DX is fixed to Wilder (rma) smoothing over a drift=1 directional
    # move and emits a 0-100 index; only route there when those match so a
    # non-default mamode/drift falls back to the native path (which honors them).
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1 and _mamode == "rma" and drift == 1
    _length = length
    _scalar = scalar

    if _use_talib:

        def compute_dx(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            h = df["_h"].to_numpy().astype(np.float64)
            l_ = df["_l"].to_numpy().astype(np.float64)
            c = df["_c"].to_numpy().astype(np.float64)
            from talib import DX as TALIB_DX

            result = TALIB_DX(h, l_, c, timeperiod=_length)
            if _scalar != 100.0:
                # TA-Lib DX is 0-100; rescale so the user's scalar is honored.
                result = result * (_scalar / 100.0)
            return pl.Series(result)

        struct_expr = pl.struct(high_expr.alias("_h"), low_expr.alias("_l"), close_expr.alias("_c"))
        dx_expr = struct_expr.map_batches(compute_dx, return_dtype=pl.Float64)
    else:
        up = high_expr - high_expr.shift(drift)
        dn = low_expr.shift(drift) - low_expr

        pos_raw = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
        neg_raw = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)

        # SMA-seeded (presma) RMA matches the classic reference and TA-Lib.
        _ma_kwargs = {"presma": True} if _mamode == "rma" else {}
        dmp = ma(_mamode, pos_raw, length=length, offset=0, **_ma_kwargs)
        dmn = ma(_mamode, neg_raw, length=length, offset=0, **_ma_kwargs)

        dx_expr = scalar * (dmp - dmn).abs() / (dmp + dmn)

    if offset != 0:
        dx_expr = dx_expr.shift(offset)

    return dx_expr.alias(f"DX_{length}")
