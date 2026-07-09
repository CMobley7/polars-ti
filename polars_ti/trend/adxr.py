# -*- coding: utf-8 -*-
# =============================================================================
# Polars ADXR (Average Directional Movement Index Rating) Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def adxr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    lensig: int | None = None,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Average Directional Movement Index Rating (ADXR)

    ADXR smooths the ADX by averaging the current ADX with the ADX value from
    ``length - 1`` bars ago, giving a slower-moving read on trend strength.

    ADXR = (ADX + ADX.shift(length - 1)) / 2

    The ``length - 1`` lookback matches TA-Lib's ``ADXR`` definition exactly
    (TA-Lib averages ADX[i] with ADX[i - (timeperiod - 1)]).

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: ADX/DM period. Default: 14
        lensig: ADX signal (smoothing) period. Defaults to ``length`` when None.
        scalar: Magnification factor. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ADXR expression aliased ``ADXR_{lensig}``
    """
    from polars_ti.maps import Imports
    from polars_ti.trend.adx import adx
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if high_expr is None or low_expr is None or close_expr is None:
        return None

    # TA-Lib's ADXR exposes only a single ``timeperiod``, so its fast path is
    # valid only when the ADX smoothing period equals ``length``. lensig defaults
    # to length when omitted (matching pandas-ta), keeping the fast path active
    # for the common case; an explicit lensig != length routes to native.
    _lensig = lensig if lensig is not None else length
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1 and _lensig == length
    _length = length

    if _use_talib:

        def compute_adxr(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            h = df["_h"].to_numpy().astype(np.float64)
            l_ = df["_l"].to_numpy().astype(np.float64)
            c = df["_c"].to_numpy().astype(np.float64)
            from talib import ADXR as TALIB_ADXR

            return pl.Series(TALIB_ADXR(h, l_, c, timeperiod=_length))

        struct_expr = pl.struct(high_expr.alias("_h"), low_expr.alias("_l"), close_expr.alias("_c"))
        adxr_expr = struct_expr.map_batches(compute_adxr, return_dtype=pl.Float64)
    else:
        adx_line = adx(
            high_expr,
            low_expr,
            close_expr,
            length=length,
            lensig=_lensig,
            scalar=scalar,
            talib=False,
        ).struct.field(f"ADX_{_lensig}")
        adxr_expr = 0.5 * (adx_line + adx_line.shift(length - 1))

    if offset != 0:
        adxr_expr = adxr_expr.shift(offset)

    return adxr_expr.alias(f"ADXR_{_lensig}")
