# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_PATTERN Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles.cdl_doji import pl_cdl_doji
from polars_ti.candles.cdl_inside import pl_cdl_inside


# Full list of TA-Lib candle patterns (used when talib=True)
ALL_PATTERNS = [
    "2crows", "3blackcrows", "3inside", "3linestrike", "3outside",
    "3starsinsouth", "3whitesoldiers", "abandonedbaby", "advanceblock",
    "belthold", "breakaway", "closingmarubozu", "concealbabyswall",
    "counterattack", "darkcloudcover", "doji", "dojistar", "dragonflydoji",
    "engulfing", "eveningdojistar", "eveningstar", "gapsidesidewhite",
    "gravestonedoji", "hammer", "hangingman", "harami", "haramicross",
    "highwave", "hikkake", "hikkakemod", "homingpigeon", "identical3crows",
    "inneck", "inside", "invertedhammer", "kicking", "kickingbylength",
    "ladderbottom", "longleggeddoji", "longline", "marubozu", "matchinglow",
    "mathold", "morningdojistar", "morningstar", "onneck", "piercing",
    "rickshawman", "risefall3methods", "separatinglines", "shootingstar",
    "shortline", "spinningtop", "stalledpattern", "sticksandwich", "takuri",
    "tasukigap", "thrusting", "tristar", "unique3river", "upsidegap2crows",
    "xsidegap3methods",
]

# Polars-native patterns available
POLARS_PATTERNS = {
    "doji": pl_cdl_doji,
    "inside": pl_cdl_inside,
}


def pl_cdl_pattern(
    df: pl.DataFrame,
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    name: str | list[str] = "all",
    talib: bool = True,
    scalar: float = 100.0,
    offset: int = 0,
) -> pl.DataFrame:
    """Polars: Candlestick Pattern Detection

    Detects candlestick patterns using native Polars implementations or TA-Lib.
    Native Polars: doji, inside
    TA-Lib (when installed and talib=True): all 61 patterns

    Args:
        df: Polars DataFrame with OHLC columns
        open_: Column name for 'open' prices. Default: "open"
        high: Column name for 'high' prices. Default: "high"
        low: Column name for 'low' prices. Default: "low"
        close: Column name for 'close' prices. Default: "close"
        name: Pattern name(s) to detect. Default: "all"
        talib: If True and TA-Lib is installed, uses TA-Lib for patterns. Default: True
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.DataFrame: Original DataFrame with pattern columns added
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    _use_talib = Imports["talib"] and v_talib(talib)

    
    if name == "all":
        if _use_talib:
            patterns = ALL_PATTERNS
        else:
            patterns = list(POLARS_PATTERNS.keys())
    elif isinstance(name, str):
        patterns = [name]
    else:
        patterns = list(name)

    result_df = df
    
    for pattern in patterns:
        # Try native Polars first
        if pattern in POLARS_PATTERNS:
            if pattern == "doji":
                expr = pl_cdl_doji(open_, high, low, close, scalar=scalar, offset=offset)
            elif pattern == "inside":
                expr = pl_cdl_inside(open_, high, low, close, scalar=scalar, offset=offset)
            result_df = result_df.with_columns(expr)
        elif _use_talib and pattern in ALL_PATTERNS:
            # Use TA-Lib via map_batches
            import talib.abstract as tala
            
            _pattern = pattern
            _scalar = scalar
            _offset = offset
            
            def compute_pattern(struct: pl.Series) -> pl.Series:
                # Extract OHLC from struct
                o = struct.struct.field("o").to_numpy().astype(np.float64)
                h = struct.struct.field("h").to_numpy().astype(np.float64)
                l = struct.struct.field("l").to_numpy().astype(np.float64)
                c = struct.struct.field("c").to_numpy().astype(np.float64)
                
                pf = tala.Function(f"CDL{_pattern.upper()}")
                result = 0.01 * _scalar * pf(o, h, l, c)
                
                if _offset != 0:
                    result = np.roll(result, _offset)
                    if _offset > 0:
                        result[:_offset] = np.nan
                    else:
                        result[_offset:] = np.nan
                
                return pl.Series(result)

            # Create struct with OHLC, then apply pattern detection
            struct_expr = pl.struct([
                pl.col(open_).alias("o"),
                pl.col(high).alias("h"),
                pl.col(low).alias("l"),
                pl.col(close).alias("c"),
            ])
            
            pattern_result = df.select(
                struct_expr.map_batches(compute_pattern, return_dtype=pl.Float64).alias(f"CDL_{pattern.upper()}")
            )
            result_df = result_df.with_columns(pattern_result)
        else:
            print(f"[X] Pattern '{pattern}' not available. Install TA-Lib for: {pattern}")

    return result_df


def pl_cdl_doji_expr(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
) -> PlExpr:
    """Alias for pl_cdl_doji for consistency."""
    return pl_cdl_doji(open_, high, low, close, length, factor, scalar)


def pl_cdl_inside_expr(
    high: IntoExpr,
    low: IntoExpr,
    scalar: float = 100.0,
) -> PlExpr:
    """Alias for pl_cdl_inside for consistency."""
    return pl_cdl_inside(high, low, scalar)


pl_cdl = pl_cdl_pattern  # Alias matching pandas naming convention

