# -*- coding: utf-8 -*-
"""Tests for pl_ichimoku."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.ichimoku import pl_ichimoku


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

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 102 + np.cumsum(np.random.randn(n) * 0.3)
        low = 98 + np.cumsum(np.random.randn(n) * 0.3)
        close = 100 + np.cumsum(np.random.randn(n) * 0.4)
        return {
            "pd_high": high,
            "pd_low": low,
            "pd_close": close,
            "pl_df": pl.DataFrame({"high": high, "low": low, "close": close}),
        }

    def test_returns_two_dataframes(self, sample_df):
        main_df, span_df = pl_ichimoku(sample_df)
        assert isinstance(main_df, pl.DataFrame)
        assert isinstance(span_df, pl.DataFrame)

    def test_main_df_columns(self, sample_df):
        main_df, _ = pl_ichimoku(sample_df)
        assert "ISA_9" in main_df.columns
        assert "ISB_26" in main_df.columns
        assert "ITS_9" in main_df.columns
        assert "IKS_26" in main_df.columns
        assert "ICS_26" in main_df.columns

    def test_exclude_chikou(self, sample_df):
        main_df, _ = pl_ichimoku(sample_df, include_chikou=False)
        assert "ICS_26" not in main_df.columns
        assert len(main_df.columns) == 4

    def test_span_df_columns(self, sample_df):
        _, span_df = pl_ichimoku(sample_df)
        assert "ISA_9" in span_df.columns
        assert "ISB_26" in span_df.columns

    def test_custom_periods(self, sample_df):
        main_df, _ = pl_ichimoku(sample_df, tenkan=7, kijun=22, senkou=44)
        assert "ISA_7" in main_df.columns
        assert "ISB_22" in main_df.columns
        assert "ITS_7" in main_df.columns
        assert "IKS_22" in main_df.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame(
            {
                "high": [None] + [102.0] * 79,
                "low": [None] + [98.0] * 79,
                "close": [None] + [100.0] * 79,
            }
        )
        main_df, span_df = pl_ichimoku(df)
        assert main_df.height == 80

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame(
            {
                "high": [0.0] * 5 + [102.0] * 75,
                "low": [0.0] * 5 + [98.0] * 75,
                "close": [0.0] * 5 + [100.0] * 75,
            }
        )
        main_df, span_df = pl_ichimoku(df)
        assert main_df.height == 80
