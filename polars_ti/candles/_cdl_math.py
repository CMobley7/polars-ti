# -*- coding: utf-8 -*-
"""Core candle pattern framework — replicates TA-Lib's C candle macros.

Ported from ``pandas_ta_classic/candles/_cdl_math.py`` (a verbatim port of
TA-Lib's ``ta_global.c`` / ``ta_defs.h`` candle machinery) onto Polars.

The heavy lifting is done on NumPy arrays inside :func:`run_pattern` (the
per-bar sequential scans TA-Lib uses cannot be vectorised as pure Polars
expressions without diverging), which is wrapped into a ``pl.Expr`` via
``map_batches`` so it composes with the rest of the Polars-TI API.

The leading underscore keeps this file out of any category auto-discovery.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable

import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr

# ---------------------------------------------------------------------------
# Enums (mirror TA-Lib ta_defs.h)
# ---------------------------------------------------------------------------


class RangeType(IntEnum):
    RealBody = 0
    HighLow = 1
    Shadows = 2


class CandleSetting(IntEnum):
    BodyLong = 0
    BodyVeryLong = 1
    BodyShort = 2
    BodyDoji = 3
    ShadowLong = 4
    ShadowVeryLong = 5
    ShadowShort = 6
    ShadowVeryShort = 7
    Near = 8
    Far = 9
    Equal = 10


# ---------------------------------------------------------------------------
# Default settings  (range_type, avg_period, factor)
# From TA-Lib ta_global.c  TA_CandleDefaultSettings
# ---------------------------------------------------------------------------

CANDLE_DEFAULTS = {
    CandleSetting.BodyLong: (RangeType.RealBody, 10, 1.0),
    CandleSetting.BodyVeryLong: (RangeType.RealBody, 10, 3.0),
    CandleSetting.BodyShort: (RangeType.RealBody, 10, 1.0),
    CandleSetting.BodyDoji: (RangeType.HighLow, 10, 0.1),
    CandleSetting.ShadowLong: (RangeType.RealBody, 0, 1.0),
    CandleSetting.ShadowVeryLong: (RangeType.RealBody, 0, 2.0),
    CandleSetting.ShadowShort: (RangeType.Shadows, 10, 1.0),
    CandleSetting.ShadowVeryShort: (RangeType.HighLow, 10, 0.1),
    CandleSetting.Near: (RangeType.HighLow, 5, 0.2),
    CandleSetting.Far: (RangeType.HighLow, 5, 0.6),
    CandleSetting.Equal: (RangeType.HighLow, 5, 0.05),
}


# ---------------------------------------------------------------------------
# Pre-computed average parameters (module-level for direct access in _detect)
# ---------------------------------------------------------------------------

AVG_PERIOD = {s: CANDLE_DEFAULTS[s][1] for s in CandleSetting}

AVG_FACTOR: dict[CandleSetting, float] = {}
for _s in CandleSetting:
    _rt, _ap, _f = CANDLE_DEFAULTS[_s]
    _d = 2.0 if _rt == RangeType.Shadows else 1.0
    AVG_FACTOR[_s] = _f / (_ap * _d) if _ap != 0 else _f / _d
del _s, _rt, _ap, _f, _d


# ---------------------------------------------------------------------------
# CandleArrays — pre-computed numpy arrays + TA-Lib macro equivalents
# ---------------------------------------------------------------------------


class CandleArrays:
    """Holds pre-computed OHLC-derived numpy arrays and provides TA-Lib
    macro equivalents (``candle_range``, ``candle_average``, etc.)."""

    __slots__ = (
        "_ranges",
        "body_high",
        "body_low",
        "close",
        "color",
        "high",
        "hl_range",
        "low",
        "lower_shadow",
        "open",
        "real_body",
        "shadow_range",
        "upper_shadow",
    )

    def __init__(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> None:
        self.open = open_
        self.high = high
        self.low = low
        self.close = close

        self.body_high = np.maximum(close, open_)
        self.body_low = np.minimum(close, open_)
        self.real_body = np.abs(close - open_)
        self.upper_shadow = high - self.body_high
        self.lower_shadow = self.body_low - low
        self.hl_range = high - low
        self.shadow_range = self.upper_shadow + self.lower_shadow
        # +1 = bullish (close >= open), -1 = bearish
        self.color = np.where(close >= open_, 1, -1)

        # Pre-computed range array for each CandleSetting (eliminates
        # per-call branching in candle_range).
        _rt_arrays = {
            RangeType.RealBody: self.real_body,
            RangeType.HighLow: self.hl_range,
            RangeType.Shadows: self.shadow_range,
        }
        self._ranges = {s: _rt_arrays[CANDLE_DEFAULTS[s][0]] for s in CandleSetting}

    # -- TA-Lib macro: TA_CANDLERANGE --
    def candle_range(self, setting: CandleSetting, i: int) -> float:
        return self._ranges[setting][i]

    # -- TA-Lib macro: TA_CANDLEAVERAGE --
    def candle_average(self, setting: CandleSetting, period_total: float, i: int) -> float:
        """Exact replica of TA-Lib's TA_CANDLEAVERAGE macro."""
        ap = AVG_PERIOD[setting]
        if ap != 0:
            return AVG_FACTOR[setting] * period_total
        return AVG_FACTOR[setting] * self._ranges[setting][i]

    # -- TA-Lib macro: TA_REALBODYGAPUP --
    def real_body_gap_up(self, i2: int, i1: int) -> bool:
        return self.body_low[i2] > self.body_high[i1]

    # -- TA-Lib macro: TA_REALBODYGAPDOWN --
    def real_body_gap_down(self, i2: int, i1: int) -> bool:
        return self.body_high[i2] < self.body_low[i1]

    # -- TA-Lib macro: TA_CANDLEGAPUP --
    def candle_gap_up(self, i2: int, i1: int) -> bool:
        return self.low[i2] > self.high[i1]

    # -- TA-Lib macro: TA_CANDLEGAPDOWN --
    def candle_gap_down(self, i2: int, i1: int) -> bool:
        return self.high[i2] < self.low[i1]


# ---------------------------------------------------------------------------
# Lookback helper
# ---------------------------------------------------------------------------


def candle_avg_period(setting: CandleSetting) -> int:
    return CANDLE_DEFAULTS[setting][1]


# ---------------------------------------------------------------------------
# run_pattern — top-level helper that every cdl_*.py calls
# ---------------------------------------------------------------------------


def run_pattern(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    detect_fn: Callable[..., None],
    name: str,
    talib_name: str,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
    **kwargs: Any,
) -> PlExpr:
    """Build a ``pl.Expr`` that computes a candlestick pattern.

    Args:
        open_, high, low, close: Column names or ``pl.Expr`` for OHLC prices.
        detect_fn: ``fn(ca: CandleArrays, out: np.ndarray, **kwargs)`` that
            fills *out* in place with pattern signals (100 / -100 / 0).
        name: Output column name, e.g. ``"CDL_ENGULFING"``.
        talib_name: TA-Lib function suffix, e.g. ``"ENGULFING"`` for
            ``talib.CDLENGULFING``.
        scalar: Multiplier for output values (TA-Lib emits ±100). Default: 100.
        offset: How many periods to shift the result. Default: 0.
        talib: When True and TA-Lib is installed, use ``talib.CDL<NAME>``.
            Default: True.
        **kwargs: Forwarded to *detect_fn* (e.g. ``penetration``).

    Returns:
        ``pl.Expr`` producing the pattern column (Float64, ±scalar / 0).
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _use_talib = bool(Imports["talib"] and v_talib(talib))
    _scalar = float(scalar) if scalar else 100.0
    _offset = offset
    _kwargs = dict(kwargs)

    def compute(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        o = df["_open"].to_numpy().astype(np.float64)
        h = df["_high"].to_numpy().astype(np.float64)
        lo = df["_low"].to_numpy().astype(np.float64)
        c = df["_close"].to_numpy().astype(np.float64)

        if _use_talib:
            import talib

            fn = getattr(talib, f"CDL{talib_name}")
            # Penetration-parametrised TA-Lib functions take it as a kwarg.
            if "penetration" in _kwargs:
                out = fn(o, h, lo, c, penetration=_kwargs["penetration"])
            else:
                out = fn(o, h, lo, c)
            out = out.astype(np.float64)
        else:
            ca = CandleArrays(o, h, lo, c)
            out = np.zeros(c.shape[0], dtype=np.float64)
            detect_fn(ca, out, **_kwargs)

        # Scale (TA-Lib emits ±100; scalar lets callers adjust).
        if _scalar != 100.0:
            mask = out != 0
            out[mask] = out[mask] / 100.0 * _scalar

        if _offset != 0:
            out = np.roll(out, _offset)
            if _offset > 0:
                out[:_offset] = np.nan
            else:
                out[_offset:] = np.nan

        return pl.Series(out)

    struct_expr = pl.struct(
        [
            open_expr.alias("_open"),
            high_expr.alias("_high"),
            low_expr.alias("_low"),
            close_expr.alias("_close"),
        ]
    )
    return struct_expr.map_batches(compute, return_dtype=pl.Float64).alias(name)
