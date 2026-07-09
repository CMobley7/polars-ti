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

    def test_forward_span_restored(self, sample_df):
        """Regression: restored OLD forward-projection ("future cloud") spans.

        Default (forward=False) is unchanged, while forward=True adds the
        forward-projected Span A/B columns without altering the existing ones.
        """
        default = _ichimoku(sample_df)
        assert "ISA_9_F" not in default.columns
        assert "ISB_26_F" not in default.columns

        fwd = _ichimoku(sample_df, forward=True)
        assert "ISA_9_F" in fwd.columns
        assert "ISB_26_F" in fwd.columns

        # Existing columns are byte-identical between the two calls.
        for col in ("ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"):
            a = default[col].to_numpy()
            b = fwd[col].to_numpy()
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            m = ~np.isnan(a)
            assert np.array_equal(a[m], b[m])

        # The forward span carries the un-shifted Senkou values: projecting the
        # visible Span A back by (kijun - 1) reproduces the forward Span A.
        kijun = 26
        visible_a = fwd["ISA_9"].to_numpy()
        forward_a = fwd["ISA_9_F"].to_numpy()
        reconstructed = np.concatenate([visible_a[kijun - 1 :], np.full(kijun - 1, np.nan)])
        m = ~np.isnan(reconstructed) & ~np.isnan(forward_a)
        assert m.sum() > 0
        np.testing.assert_allclose(reconstructed[m], forward_a[m], rtol=1e-9)
