# -*- coding: utf-8 -*-
# =============================================================================
# Polars CRSI (Connors RSI) Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_consecutive_streak(close: np.ndarray) -> np.ndarray:
    """Numba: Calculate streak of consecutive price increases/decreases."""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0:
            result[i] = 1.0
        elif diff < 0:
            result[i] = -1.0
        else:
            result[i] = 0.0
    return result


@njit(cache=True)
def nb_percent_rank(close: np.ndarray, lookback: int) -> np.ndarray:
    """Numba: Calculate percent rank of daily returns over lookback period."""
    n = len(close)
    result = np.full(n, np.nan)
    
    # Calculate daily returns
    returns = np.empty(n)
    returns[0] = np.nan
    for i in range(1, n):
        returns[i] = (close[i] - close[i - 1]) / close[i - 1]
    
    # Calculate percent rank
    for i in range(lookback, n):
        current = returns[i]
        count_less = 0
        valid_count = 0
        for j in range(i - lookback, i):
            if not np.isnan(returns[j]):
                valid_count += 1
                if returns[j] < current:
                    count_less += 1
        if valid_count > 0:
            result[i] = (count_less / valid_count) * 100.0
    
    return result


def pl_crsi(
    close: IntoExpr,
    length_rsi: int = 3,
    length_streak: int = 2,
    length_rank: int = 100,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Connors Relative Strength Index (CRSI)

    Integrates RSI, UpDown Streak RSI, and Percent Rank to evaluate 
    overbought/oversold conditions.
    Formula: CRSI = (RSI(close) + RSI(streak) + PercentRank) / 3

    Args:
        close: Column name or pl.Expr for 'close' prices
        length_rsi: RSI period. Default: 3
        length_streak: Streak RSI period. Default: 2
        length_rank: Percent Rank lookback. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib for RSI. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CRSI expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.momentum.rsi import pl_rsi
    
    close_expr = v_expr(close)
    _use_talib = Imports["talib"] and v_talib(talib)
    
    _length_rsi = length_rsi
    _length_streak = length_streak
    _length_rank = length_rank
    
    # Compute streak as expression: sign of price diff
    def compute_streak(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        return pl.Series(nb_consecutive_streak(arr))
    
    streak_expr = close_expr.map_batches(compute_streak, return_dtype=pl.Float64)
    
    # Compute percent rank using Numba
    def compute_pr(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        return pl.Series(nb_percent_rank(arr, _length_rank))
    
    pr_expr = close_expr.map_batches(compute_pr, return_dtype=pl.Float64)
    
    # Use pl_rsi for both RSI calculations (reuses existing RSI implementation)
    close_rsi_expr = pl_rsi(close_expr, length=length_rsi, talib=talib, offset=0)
    streak_rsi_expr = pl_rsi(streak_expr, length=length_streak, talib=talib, offset=0)
    
    # CRSI = (close_rsi + streak_rsi + percent_rank) / 3
    crsi_expr = (close_rsi_expr + streak_rsi_expr + pr_expr) / 3.0
    
    if offset != 0:
        crsi_expr = crsi_expr.shift(offset)
    
    return crsi_expr.alias(f"CRSI_{length_rsi}_{length_streak}_{length_rank}")

