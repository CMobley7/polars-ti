# -*- coding: utf-8 -*-
"""Tests for ichimoku."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.ichimoku import ichimoku


def _ichimoku(df, **kwargs):
    """Helper: run ichimoku and return the unnested struct as a flat DataFrame."""
    result = df.select(ichimoku("high", "low", "close", **kwargs))
    return result.unnest(result.columns[0])


class TestPlIchimoku:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "high": 102 + np.cumsum(np.random.randn(n) * 0.3),
                "low": 98 + np.cumsum(np.random.randn(n) * 0.3),
                "close": 100 + np.cumsum(np.random.randn(n) * 0.4),
            }
        )

    def test_returns_expression(self, sample_df):
        assert isinstance(ichimoku("high", "low", "close"), pl.Expr)

    def test_main_df_columns(self, sample_df):
        result = _ichimoku(sample_df)
        assert "ISA_9" in result.columns
        assert "ISB_26" in result.columns
        assert "ITS_9" in result.columns
        assert "IKS_26" in result.columns
        assert "ICS_26" in result.columns

    def test_exclude_chikou(self, sample_df):
        result = _ichimoku(sample_df, include_chikou=False)
        assert "ICS_26" not in result.columns
        assert len(result.columns) == 4

    def test_lookahead_false_excludes_chikou(self, sample_df):
        result = _ichimoku(sample_df, lookahead=False)
        assert "ICS_26" not in result.columns

    def test_custom_periods(self, sample_df):
        result = _ichimoku(sample_df, tenkan=7, kijun=22, senkou=44)
        assert "ISA_7" in result.columns
        assert "ISB_22" in result.columns
        assert "ITS_7" in result.columns
        assert "IKS_22" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [102.0] * 79,
                "low": [None] + [98.0] * 79,
                "close": [None] + [100.0] * 79,
            }
        )
        result = _ichimoku(df)
        assert result.height == 80

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [102.0] * 75,
                "low": [0.0] * 5 + [98.0] * 75,
                "close": [0.0] * 5 + [100.0] * 75,
            }
        )
        result = _ichimoku(df)
        assert result.height == 80
