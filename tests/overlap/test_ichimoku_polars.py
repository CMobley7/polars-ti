# -*- coding: utf-8 -*-
"""Tests for pl_ichimoku."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest
from polars_ti.overlap.ichimoku import pl_ichimoku


class TestPlIchimoku:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            'high': 102 + np.cumsum(np.random.randn(n) * 0.3),
            'low': 98 + np.cumsum(np.random.randn(n) * 0.3),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.4),
        })

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        high = 102 + np.cumsum(np.random.randn(n) * 0.3)
        low = 98 + np.cumsum(np.random.randn(n) * 0.3)
        close = 100 + np.cumsum(np.random.randn(n) * 0.4)
        return {
            'pd_high': pd.Series(high),
            'pd_low': pd.Series(low),
            'pd_close': pd.Series(close),
            'pl_df': pl.DataFrame({'high': high, 'low': low, 'close': close}),
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

    def test_numerical_parity(self, sample_data):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")

        pd_main, _ = ichimoku(sample_data['pd_high'], sample_data['pd_low'], sample_data['pd_close'])
        pl_main, _ = pl_ichimoku(sample_data['pl_df'])
        
        warmup = 60
        for col in ["ISA_9", "ISB_26", "ITS_9", "IKS_26"]:
            pd_vals = pd_main[col].iloc[warmup:-25].values  # Avoid edges
            pl_vals = pl_main[col][warmup:-25].to_numpy()
            
            valid = ~np.isnan(pd_vals) & ~np.isnan(pl_vals)
            if valid.sum() > 0:
                diff = np.abs(pd_vals[valid] - pl_vals[valid])
                assert np.max(diff) < 1e-10, f"Max diff for {col}: {np.max(diff)}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({
            "high": [None] + [102.0] * 79,
            "low": [None] + [98.0] * 79,
            "close": [None] + [100.0] * 79,
        })
        main_df, span_df = pl_ichimoku(df)
        assert main_df.height == 80

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({
            "high": [0.0] * 5 + [102.0] * 75,
            "low": [0.0] * 5 + [98.0] * 75,
            "close": [0.0] * 5 + [100.0] * 75,
        })
        main_df, span_df = pl_ichimoku(df)
        assert main_df.height == 80

