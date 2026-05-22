# -*- coding: utf-8 -*-
# =============================================================================
# Polars TMO Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _signed_rolling_deltas_numba(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    length: int,
    exclusive: bool,
) -> np.ndarray:
    """Numba kernel for signed rolling deltas."""
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)

    lookback = length if exclusive else length - 1

    for i in range(lookback, n):
        sum_signed = 0.0
        for j in range(lookback):
            idx = i - lookback + j
            if idx >= 0:
                diff = close_arr[i] - open_arr[idx]
                if diff > 0:
                    sum_signed += 1.0
                elif diff < 0:
                    sum_signed -= 1.0
        result[i] = sum_signed

    return result


@njit(cache=True)
def _ema_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba EMA with presma initialization."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    # Find first valid index
    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1 or n - first_valid < length:
        return result

    # Calculate initial SMA for EMA seed
    sma_sum = 0.0
    for i in range(first_valid, first_valid + length):
        sma_sum += values[i]
    sma_val = sma_sum / length

    result[first_valid + length - 1] = sma_val

    alpha = 2.0 / (length + 1)
    for i in range(first_valid + length, n):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result


@njit(cache=True)
def _tmo_core(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    tmo_length: int,
    calc_length: int,
    smooth_length: int,
    exclusive: bool,
    compute_momentum: bool,
) -> tuple:
    """Numba kernel for TMO calculation."""
    n = len(close_arr)

    # 1. Calculate signed rolling deltas
    signed_diff = _signed_rolling_deltas_numba(open_arr, close_arr, tmo_length, exclusive)

    # 2. Initial MA smoothing
    initial_ma = _ema_numba(signed_diff, calc_length)

    # 3. Main signal = EMA(initial_ma, smooth_length)
    main = _ema_numba(initial_ma, smooth_length)

    # 4. Smooth signal = EMA(main, smooth_length)
    smooth = _ema_numba(main, smooth_length)

    # 5. Momentum (if requested)
    if compute_momentum:
        mom_main = np.full(n, np.nan, dtype=np.float64)
        mom_smooth = np.full(n, np.nan, dtype=np.float64)
        for i in range(tmo_length, n):
            if not np.isnan(main[i]) and not np.isnan(main[i - tmo_length]):
                mom_main[i] = main[i] - main[i - tmo_length]
            if not np.isnan(smooth[i]) and not np.isnan(smooth[i - tmo_length]):
                mom_smooth[i] = smooth[i] - smooth[i - tmo_length]
    else:
        mom_main = np.zeros(n, dtype=np.float64)
        mom_smooth = np.zeros(n, dtype=np.float64)

    return main, smooth, mom_main, mom_smooth


def pl_tmo(
    open_: IntoExpr,
    close: IntoExpr,
    tmo_length: int = 14,
    calc_length: int = 5,
    smooth_length: int = 3,
    momentum: bool = False,
    normalize: bool = False,
    exclusive: bool = True,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: True Momentum Oscillator (TMO)

    The True Momentum Oscillator measures the momentum of an asset's price
    movement over a specified time frame by comparing closing to opening
    prices within a rolling window, summing the sign of differences, and
    applying moving averages to smooth the results.

    Sources:
        https://www.tradingview.com/script/VRwDppqd-True-Momentum-Oscillator/
        https://www.tradingview.com/script/65vpO7T5-True-Momentum-Oscillator-Universal-Edition/

    Args:
        open_: Column name or pl.Expr for 'open' prices
        close: Column name or pl.Expr for 'close' prices
        tmo_length: Period for TMO calculation. Default: 14
        calc_length: Initial moving average window. Default: 5
        smooth_length: Main and smooth signal MA window. Default: 3
        momentum: Compute main and smooth momentum. Default: False
        normalize: Normalize TMO values to [-100, 100]. Default: False
        exclusive: Exclusive rolling window (True) or inclusive. Default: True
        mamode: MA type for smoothing. Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - TMO_{tmo_length}_{calc_length}_{smooth_length}: Main signal
            - TMOs_{tmo_length}_{calc_length}_{smooth_length}: Smooth signal
            - TMOM_{tmo_length}_{calc_length}_{smooth_length}: Main momentum
            - TMOMs_{tmo_length}_{calc_length}_{smooth_length}: Smooth momentum
    """
    open_expr = v_expr(open_)
    close_expr = v_expr(close)

    if open_expr is None or close_expr is None:
        return None

    _tmo_length = tmo_length
    _calc_length = calc_length
    _smooth_length = smooth_length
    _exclusive = exclusive
    _compute_momentum = momentum
    _normalize = normalize
    _props = f"_{tmo_length}_{calc_length}_{smooth_length}"

    def compute_tmo(s: pl.Series) -> pl.Series:
        open_arr = s.struct.field("open").to_numpy().astype(np.float64)
        close_arr = s.struct.field("close").to_numpy().astype(np.float64)

        main, smooth, mom_main, mom_smooth = _tmo_core(
            open_arr,
            close_arr,
            _tmo_length,
            _calc_length,
            _smooth_length,
            _exclusive,
            _compute_momentum,
        )

        # Normalize if requested
        if _normalize:
            max_val = _tmo_length
            main = 100.0 * main / max_val
            smooth = 100.0 * smooth / max_val
            if _compute_momentum:
                mom_main = 100.0 * mom_main / max_val
                mom_smooth = 100.0 * mom_smooth / max_val

        return pl.DataFrame(
            {
                f"TMO{_props}": main,
                f"TMOs{_props}": smooth,
                f"TMOM{_props}": mom_main,
                f"TMOMs{_props}": mom_smooth,
            }
        ).to_struct("TMO")

    result_expr = pl.struct(
        open=open_expr,
        close=close_expr,
    ).map_batches(
        compute_tmo,
        return_dtype=pl.Struct(
            [
                pl.Field(f"TMO{_props}", pl.Float64),
                pl.Field(f"TMOs{_props}", pl.Float64),
                pl.Field(f"TMOM{_props}", pl.Float64),
                pl.Field(f"TMOMs{_props}", pl.Float64),
            ]
        ),
    )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("TMO")
