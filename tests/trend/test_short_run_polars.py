# -*- coding: utf-8 -*-
"""Tests for pl_short_run."""
import numpy as np
import polars as pl
import pytest
from polars_ti.trend.short_run import pl_short_run


def _np_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Pure NumPy EWM equivalent for test fixtures."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


class TestPlShortRun:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        # Create fast and slow EMAs
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pl.DataFrame({
            'fast': _np_ema(close, span=5),
            'slow': _np_ema(close, span=20),
        })

    def test_returns_expression(self, sample_df):
        result = sample_df.select(pl_short_run("fast", "slow"))
        assert result.height == 100

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_short_run("fast", "slow", length=3))
        assert "SR_3" in result.columns

    def test_offset_shifts_result(self, sample_df):
        result = sample_df.select(pl_short_run("fast", "slow", offset=5))
        arr = result[result.columns[0]].to_numpy()
        assert all(v is None or np.isnan(float(v)) for v in arr[:5] if v is not None)

    def test_with_null_values(self):
        df = pl.DataFrame({
            "fast": [None] + [100.0] * 49,
            "slow": [None] + [100.0] * 49
        })
        result = df.select(pl_short_run("fast", "slow", length=2))
        assert result.height == 50

    def test_with_zeros(self):
        df = pl.DataFrame({
            "fast": [0.0] * 50,
            "slow": [0.0] * 50
        })
        result = df.select(pl_short_run("fast", "slow", length=2))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_short_run("fast", "slow", length=2)).collect()
        assert "SR_2" in result.columns
