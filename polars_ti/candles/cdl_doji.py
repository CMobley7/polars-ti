# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_DOJI Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils import v_talib
from polars_ti.utils._candles import high_low_range, real_body
from polars_ti.utils._validate import v_expr


def cdl_doji(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
    asint: bool = True,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Candle Type - Doji

    A candle body is Doji when it's shorter than a percentage of
    the average of the previous candles' high-low range.

    Sources:
        TA-Lib: 96.56% Correlation

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: The averaging period. Default: 10
        factor: Doji threshold percentage. Default: 10.0 (means 10%)
        scalar: Result multiplier. Default: 100.0
        asint: Return integer (scaled) instead of boolean. Default: True
        talib: If True and TA-Lib is installed, route via talib.CDLDOJI.
            Default: True. NOTE (downgrade): talib.CDLDOJI exposes no
            ``length``/``factor`` knobs — it uses TA-Lib's built-in candle
            settings — so on the TA-Lib path ``length`` and ``factor`` are
            ignored (they still name the output column for consistency). The
            native default is pinned to equal ``talib.CDLDOJI`` on the shared
            (post-warmup) region, so the default output is unchanged apart from
            TA-Lib emitting 0 during its lookback where the native path emits
            null.
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CDL_DOJI expression (scalar for doji, 0 otherwise)
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # Consistency with hl2: a None input expr yields None rather than raising.
    if any(e is None for e in [open_expr, high_expr, low_expr, close_expr]):
        return None

    _alias = f"CDL_DOJI_{length}_{0.01 * factor}"

    if Imports["talib"] and v_talib(talib):
        _scalar = scalar
        _asint = asint

        def _compute(s: pl.Series) -> pl.Series:
            from talib import CDLDOJI

            data = s.struct.unnest()
            raw = CDLDOJI(
                data["_o"].to_numpy().astype(np.float64),
                data["_h"].to_numpy().astype(np.float64),
                data["_l"].to_numpy().astype(np.float64),
                data["_c"].to_numpy().astype(np.float64),
            )
            hit = raw != 0
            if _asint:
                return pl.Series(np.where(hit, _scalar, 0.0).astype(np.int64))
            return pl.Series(hit)

        doji = pl.struct(
            open_expr.alias("_o"),
            high_expr.alias("_h"),
            low_expr.alias("_l"),
            close_expr.alias("_c"),
        ).map_batches(_compute, return_dtype=pl.Int64 if asint else pl.Boolean)

        if offset != 0:
            doji = doji.shift(offset)

        return doji.alias(_alias)

    # Calculate real body (absolute difference between close and open)
    body = (close_expr - open_expr).abs()

    # Calculate high-low range
    hl_range = (high_expr - low_expr).abs()

    # Calculate average high-low range over the *previous* bars. The average is
    # shifted by one bar (classic fork commit 9258bf6) so the current body is
    # measured against yesterday's average range, not today's (avoids look-ahead
    # and matches TA-Lib's CDLDOJI).
    hl_range_avg = hl_range.rolling_mean(window_size=length, min_samples=length).shift(1)

    # Doji: body <= 0.01 * factor * average HL range. The comparison is "<=" so
    # zero-body bars count as doji (matches TA-Lib).
    threshold = 0.01 * factor * hl_range_avg

    if asint:
        doji = pl.when(body <= threshold).then(scalar).otherwise(0.0).cast(pl.Int64)
    else:
        doji = body <= threshold

    # Apply offset
    if offset != 0:
        doji = doji.shift(offset)

    return doji.alias(_alias)
