# -*- coding: utf-8 -*-
# =============================================================================
# Polars Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def squeeze(
    high: IntoExpr = "high",
    low: IntoExpr = "low",
    close: IntoExpr = "close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    kc_length: int = 20,
    kc_scalar: float = 1.5,
    mom_length: int = 12,
    mom_smooth: int = 6,
    mamode: str = "sma",
    use_tr: bool = True,
    lazybear: bool = False,
    asint: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Squeeze (SQZ)

    The default is based on John Carter's "TTM Squeeze" indicator, as
    discussed in his book "Mastering the Trade" (chapter 11). The Squeeze
    indicator captures the relationship between Bollinger Bands® and
    Keltner's Channels.

    Sources:
        https://tradestation.tradingappstore.com/products/TTMSqueeze
        https://www.tradingview.com/scripts/lazybear/
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:indicators_ttm_squeeze

    Args:
        high (IntoExpr): Column name or expression for 'high'. Default: "high"
        low (IntoExpr): Column name or expression for 'low'. Default: "low"
        close (IntoExpr): Column name or expression for 'close'. Default: "close"
        bb_length (int): Bollinger Bands period. Default: 20
        bb_std (float): Bollinger Bands Std. Dev. Default: 2
        kc_length (int): Keltner Channel period. Default: 20
        kc_scalar (float): Keltner Channel scalar. Default: 1.5
        mom_length (int): Momentum Period. Default: 12
        mom_smooth (int): Smoothing Period of Momentum. Default: 6
        mamode (str): Only "ema" or "sma". Default: "sma"
        use_tr (bool): Use True Range for Keltner Channels. Default: True
        lazybear (bool): Use LazyBear's TradingView implementation. Default: False
        asint (bool): Use integers instead of bool. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pl.Expr: Struct expression with SQZ (momentum), SQZ_ON, SQZ_OFF, SQZ_NO columns
    """
    from polars_ti.volatility.bbands import bbands
    from polars_ti.volatility.kc import kc
    from polars_ti.momentum.mom import mom
    from polars_ti.overlap.linreg import linreg
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""

    # Calculate Bollinger Bands - get struct fields
    bb_struct = bbands(close_expr, length=bb_length, std=bb_std, talib=False, offset=0)
    bb_lower_name = f"BBL_{bb_length}_{bb_std}"
    bb_upper_name = f"BBU_{bb_length}_{bb_std}"

    # Calculate Keltner Channels - get struct fields
    kc_struct = kc(
        high_expr,
        low_expr,
        close_expr,
        length=kc_length,
        scalar=kc_scalar,
        mamode=mamode,
        tr=use_tr,
        offset=0,
    )

    # We need to access struct fields, which requires using map_batches
    # to work with both inputs simultaneously
    def compute_squeeze(df_struct: pl.DataFrame) -> pl.Series:
        """Compute squeeze using struct fields."""
        # Extract BB fields
        bb_l = df_struct["bb_l"].to_numpy()
        bb_u = df_struct["bb_u"].to_numpy()

        # Extract KC fields
        kc_l = df_struct["kc_l"].to_numpy()
        kc_u = df_struct["kc_u"].to_numpy()

        # Extract momentum
        sqz_val = df_struct["sqz_val"].to_numpy()

        # Squeeze conditions
        squeeze_on = (bb_l > kc_l) & (bb_u < kc_u)
        squeeze_off = (bb_l < kc_l) & (bb_u > kc_u)
        no_squeeze = ~squeeze_on & ~squeeze_off

        if asint:
            squeeze_on = squeeze_on.astype(np.int64)
            squeeze_off = squeeze_off.astype(np.int64)
            no_squeeze = no_squeeze.astype(np.int64)

        if offset != 0:
            sqz_val = np.roll(sqz_val, offset)
            squeeze_on = np.roll(squeeze_on, offset)
            squeeze_off = np.roll(squeeze_off, offset)
            no_squeeze = np.roll(no_squeeze, offset)
            if offset > 0:
                sqz_val[:offset] = np.nan
                if asint:
                    squeeze_on[:offset] = 0
                    squeeze_off[:offset] = 0
                    no_squeeze[:offset] = 0

        return pl.DataFrame(
            {
                f"SQZ{_props}": sqz_val,
                "SQZ_ON": squeeze_on,
                "SQZ_OFF": squeeze_off,
                "SQZ_NO": no_squeeze,
            }
        ).to_struct(f"SQZ{_props}")

    # Calculate momentum component
    if lazybear:
        # LazyBear mode uses linreg
        highest_high = high_expr.rolling_max(window_size=kc_length)
        lowest_low = low_expr.rolling_min(window_size=kc_length)
        # Need kc basis (middle band)
        # For now we use simplified version
        avg_hl = (highest_high + lowest_low) / 2
        sqz_val = linreg(close_expr - avg_hl, length=kc_length, tsf=True, offset=0)
    else:
        # Standard mode: smoothed momentum
        momo = mom(close_expr, length=mom_length, talib=False, offset=0)
        sqz_val = ma(name=mamode, source=momo, length=mom_smooth, talib=False)

    # Build struct with all needed values and compute
    # This is complex - we need to handle struct extraction properly
    # We'll use a simpler approach: compute all parts separately

    # Return type depends on asint
    on_dtype = pl.Int64 if asint else pl.Boolean

    return_dtype = pl.Struct(
        [
            pl.Field(f"SQZ{_props}", pl.Float64),
            pl.Field("SQZ_ON", on_dtype),
            pl.Field("SQZ_OFF", on_dtype),
            pl.Field("SQZ_NO", on_dtype),
        ]
    )

    # Create combined struct for map_batches
    combined = pl.struct(
        [
            bb_struct.struct.field(bb_lower_name).alias("bb_l"),
            bb_struct.struct.field(bb_upper_name).alias("bb_u"),
            kc_struct.struct.field("kcl").alias("kc_l"),
            kc_struct.struct.field("kcu").alias("kc_u"),
            sqz_val.alias("sqz_val"),
        ]
    )

    return combined.map_batches(lambda s: compute_squeeze(s.struct.unnest()), return_dtype=return_dtype).alias(
        f"SQZ{_props}"
    )
