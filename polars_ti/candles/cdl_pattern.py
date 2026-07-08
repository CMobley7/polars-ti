# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_PATTERN Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles.cdl_2crows import cdl_2crows
from polars_ti.candles.cdl_3blackcrows import cdl_3blackcrows
from polars_ti.candles.cdl_3inside import cdl_3inside
from polars_ti.candles.cdl_3linestrike import cdl_3linestrike
from polars_ti.candles.cdl_3outside import cdl_3outside
from polars_ti.candles.cdl_3starsinsouth import cdl_3starsinsouth
from polars_ti.candles.cdl_3whitesoldiers import cdl_3whitesoldiers
from polars_ti.candles.cdl_abandonedbaby import cdl_abandonedbaby
from polars_ti.candles.cdl_advanceblock import cdl_advanceblock
from polars_ti.candles.cdl_belthold import cdl_belthold
from polars_ti.candles.cdl_breakaway import cdl_breakaway
from polars_ti.candles.cdl_closingmarubozu import cdl_closingmarubozu
from polars_ti.candles.cdl_concealbabyswall import cdl_concealbabyswall
from polars_ti.candles.cdl_counterattack import cdl_counterattack
from polars_ti.candles.cdl_darkcloudcover import cdl_darkcloudcover
from polars_ti.candles.cdl_doji import cdl_doji
from polars_ti.candles.cdl_dojistar import cdl_dojistar
from polars_ti.candles.cdl_dragonflydoji import cdl_dragonflydoji
from polars_ti.candles.cdl_engulfing import cdl_engulfing
from polars_ti.candles.cdl_eveningdojistar import cdl_eveningdojistar
from polars_ti.candles.cdl_eveningstar import cdl_eveningstar
from polars_ti.candles.cdl_gapsidesidewhite import cdl_gapsidesidewhite
from polars_ti.candles.cdl_gravestonedoji import cdl_gravestonedoji
from polars_ti.candles.cdl_hammer import cdl_hammer
from polars_ti.candles.cdl_hangingman import cdl_hangingman
from polars_ti.candles.cdl_harami import cdl_harami
from polars_ti.candles.cdl_haramicross import cdl_haramicross
from polars_ti.candles.cdl_highwave import cdl_highwave
from polars_ti.candles.cdl_hikkake import cdl_hikkake
from polars_ti.candles.cdl_hikkakemod import cdl_hikkakemod
from polars_ti.candles.cdl_homingpigeon import cdl_homingpigeon
from polars_ti.candles.cdl_identical3crows import cdl_identical3crows
from polars_ti.candles.cdl_inneck import cdl_inneck
from polars_ti.candles.cdl_inside import cdl_inside
from polars_ti.candles.cdl_invertedhammer import cdl_invertedhammer
from polars_ti.candles.cdl_kicking import cdl_kicking
from polars_ti.candles.cdl_kickingbylength import cdl_kickingbylength
from polars_ti.candles.cdl_ladderbottom import cdl_ladderbottom
from polars_ti.candles.cdl_longleggeddoji import cdl_longleggeddoji
from polars_ti.candles.cdl_longline import cdl_longline
from polars_ti.candles.cdl_marubozu import cdl_marubozu
from polars_ti.candles.cdl_matchinglow import cdl_matchinglow
from polars_ti.candles.cdl_mathold import cdl_mathold
from polars_ti.candles.cdl_morningdojistar import cdl_morningdojistar
from polars_ti.candles.cdl_morningstar import cdl_morningstar
from polars_ti.candles.cdl_onneck import cdl_onneck
from polars_ti.candles.cdl_piercing import cdl_piercing
from polars_ti.candles.cdl_rickshawman import cdl_rickshawman
from polars_ti.candles.cdl_risefall3methods import cdl_risefall3methods
from polars_ti.candles.cdl_separatinglines import cdl_separatinglines
from polars_ti.candles.cdl_shootingstar import cdl_shootingstar
from polars_ti.candles.cdl_shortline import cdl_shortline
from polars_ti.candles.cdl_spinningtop import cdl_spinningtop
from polars_ti.candles.cdl_stalledpattern import cdl_stalledpattern
from polars_ti.candles.cdl_sticksandwich import cdl_sticksandwich
from polars_ti.candles.cdl_takuri import cdl_takuri
from polars_ti.candles.cdl_tasukigap import cdl_tasukigap
from polars_ti.candles.cdl_thrusting import cdl_thrusting
from polars_ti.candles.cdl_tristar import cdl_tristar
from polars_ti.candles.cdl_unique3river import cdl_unique3river
from polars_ti.candles.cdl_upsidegap2crows import cdl_upsidegap2crows
from polars_ti.candles.cdl_xsidegap3methods import cdl_xsidegap3methods

# Full list of candle patterns (TA-Lib names). These are all implemented
# natively (with a TA-Lib fast-path when ``talib=True`` and TA-Lib is present),
# so both ``talib=True`` and ``talib=False`` emit the same pattern columns.
ALL_PATTERNS = [
    "2crows",
    "3blackcrows",
    "3inside",
    "3linestrike",
    "3outside",
    "3starsinsouth",
    "3whitesoldiers",
    "abandonedbaby",
    "advanceblock",
    "belthold",
    "breakaway",
    "closingmarubozu",
    "concealbabyswall",
    "counterattack",
    "darkcloudcover",
    "doji",
    "dojistar",
    "dragonflydoji",
    "engulfing",
    "eveningdojistar",
    "eveningstar",
    "gapsidesidewhite",
    "gravestonedoji",
    "hammer",
    "hangingman",
    "harami",
    "haramicross",
    "highwave",
    "hikkake",
    "hikkakemod",
    "homingpigeon",
    "identical3crows",
    "inneck",
    "inside",
    "invertedhammer",
    "kicking",
    "kickingbylength",
    "ladderbottom",
    "longleggeddoji",
    "longline",
    "marubozu",
    "matchinglow",
    "mathold",
    "morningdojistar",
    "morningstar",
    "onneck",
    "piercing",
    "rickshawman",
    "risefall3methods",
    "separatinglines",
    "shootingstar",
    "shortline",
    "spinningtop",
    "stalledpattern",
    "sticksandwich",
    "takuri",
    "tasukigap",
    "thrusting",
    "tristar",
    "unique3river",
    "upsidegap2crows",
    "xsidegap3methods",
]

# Native framework patterns: name -> cdl_<name> function. These accept the
# ``(open_, high, low, close, ..., scalar, offset, talib)`` signature and emit a
# ``CDL_<NAME>`` column. ``doji`` and ``inside`` are handled separately because
# they use the parametrised Polars implementations (``CDL_DOJI_10_0.1`` /
# ``CDL_INSIDE``).
NATIVE_PATTERNS = {
    "2crows": cdl_2crows,
    "3blackcrows": cdl_3blackcrows,
    "3inside": cdl_3inside,
    "3linestrike": cdl_3linestrike,
    "3outside": cdl_3outside,
    "3starsinsouth": cdl_3starsinsouth,
    "3whitesoldiers": cdl_3whitesoldiers,
    "abandonedbaby": cdl_abandonedbaby,
    "advanceblock": cdl_advanceblock,
    "belthold": cdl_belthold,
    "breakaway": cdl_breakaway,
    "closingmarubozu": cdl_closingmarubozu,
    "concealbabyswall": cdl_concealbabyswall,
    "counterattack": cdl_counterattack,
    "darkcloudcover": cdl_darkcloudcover,
    "dojistar": cdl_dojistar,
    "dragonflydoji": cdl_dragonflydoji,
    "engulfing": cdl_engulfing,
    "eveningdojistar": cdl_eveningdojistar,
    "eveningstar": cdl_eveningstar,
    "gapsidesidewhite": cdl_gapsidesidewhite,
    "gravestonedoji": cdl_gravestonedoji,
    "hammer": cdl_hammer,
    "hangingman": cdl_hangingman,
    "harami": cdl_harami,
    "haramicross": cdl_haramicross,
    "highwave": cdl_highwave,
    "hikkake": cdl_hikkake,
    "hikkakemod": cdl_hikkakemod,
    "homingpigeon": cdl_homingpigeon,
    "identical3crows": cdl_identical3crows,
    "inneck": cdl_inneck,
    "invertedhammer": cdl_invertedhammer,
    "kicking": cdl_kicking,
    "kickingbylength": cdl_kickingbylength,
    "ladderbottom": cdl_ladderbottom,
    "longleggeddoji": cdl_longleggeddoji,
    "longline": cdl_longline,
    "marubozu": cdl_marubozu,
    "matchinglow": cdl_matchinglow,
    "mathold": cdl_mathold,
    "morningdojistar": cdl_morningdojistar,
    "morningstar": cdl_morningstar,
    "onneck": cdl_onneck,
    "piercing": cdl_piercing,
    "rickshawman": cdl_rickshawman,
    "risefall3methods": cdl_risefall3methods,
    "separatinglines": cdl_separatinglines,
    "shootingstar": cdl_shootingstar,
    "shortline": cdl_shortline,
    "spinningtop": cdl_spinningtop,
    "stalledpattern": cdl_stalledpattern,
    "sticksandwich": cdl_sticksandwich,
    "takuri": cdl_takuri,
    "tasukigap": cdl_tasukigap,
    "thrusting": cdl_thrusting,
    "tristar": cdl_tristar,
    "unique3river": cdl_unique3river,
    "upsidegap2crows": cdl_upsidegap2crows,
    "xsidegap3methods": cdl_xsidegap3methods,
}

# Backwards-compatible alias.
POLARS_PATTERNS = {"doji": cdl_doji, "inside": cdl_inside, **NATIVE_PATTERNS}


def cdl_pattern(
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
    """Polars: Candlestick Pattern Detection.

    Detects candlestick patterns using native Polars implementations. Every
    TA-Lib candlestick pattern is implemented natively; when ``talib=True`` and
    TA-Lib is installed each pattern uses the ``talib.CDL<NAME>`` fast-path,
    otherwise the native NumPy detector runs. Both modes emit the same columns.

    Args:
        df: Polars DataFrame with OHLC columns.
        open_: Column name for 'open' prices. Default: "open"
        high: Column name for 'high' prices. Default: "high"
        low: Column name for 'low' prices. Default: "low"
        close: Column name for 'close' prices. Default: "close"
        name: Pattern name(s) to detect, or "all". Default: "all"
        talib: Use the TA-Lib fast-path when installed. Default: True
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.DataFrame: DataFrame with the requested pattern columns.
    """
    if name == "all":
        patterns = ALL_PATTERNS
    elif isinstance(name, str):
        patterns = [name]
    else:
        patterns = list(name)

    result_df = df
    for pattern in patterns:
        if pattern == "doji":
            expr = cdl_doji(open_, high, low, close, scalar=scalar, offset=offset)
        elif pattern == "inside":
            expr = cdl_inside(open_, high, low, close, scalar=scalar, offset=offset)
        elif pattern in NATIVE_PATTERNS:
            expr = NATIVE_PATTERNS[pattern](open_, high, low, close, scalar=scalar, offset=offset, talib=talib)
        else:
            print(f"[X] Pattern '{pattern}' not available. Not a known candlestick pattern.")
            continue
        result_df = result_df.with_columns(expr)

    return result_df


def cdl_doji_expr(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
) -> PlExpr:
    """Alias for cdl_doji for consistency."""
    return cdl_doji(open_, high, low, close, length, factor, scalar)


cdl = cdl_pattern  # Alias matching pandas naming convention
