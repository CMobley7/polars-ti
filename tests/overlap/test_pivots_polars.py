# -*- coding: utf-8 -*-
"""Tests for pivots."""

import datetime
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.pivots import pivots


def _pivots(df, **kwargs):
    """Helper: run pivots and return the unnested struct as a flat DataFrame."""
    result = df.select(pivots("high", "low", "close", open_="open", **kwargs))
    return result.unnest(result.columns[0])


class TestPlPivots:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 100
        start = datetime.date(2024, 1, 1)
        dates = [start + datetime.timedelta(days=i) for i in range(n)]
        return pl.DataFrame(
            {
                "date": dates,
                "open": 100 + np.cumsum(np.random.randn(n) * 0.35),
                "high": 102 + np.cumsum(np.random.randn(n) * 0.3),
                "low": 98 + np.cumsum(np.random.randn(n) * 0.3),
                "close": 100 + np.cumsum(np.random.randn(n) * 0.4),
            }
        )

    def test_returns_expression(self, sample_df):
        assert isinstance(pivots("high", "low", "close"), pl.Expr)

    def test_has_pivot_columns(self, sample_df):
        result = _pivots(sample_df)
        assert any("_P" in col for col in result.columns)
        assert any("_S1" in col for col in result.columns)
        assert any("_R1" in col for col in result.columns)

    def test_traditional_method(self, sample_df):
        result = _pivots(sample_df, method="traditional")
        assert len(result.columns) == 9  # P, S1-S4, R1-R4

    def test_fibonacci_method(self, sample_df):
        """Fibonacci emits the full P/S1-S4/R1-R4 struct; S4 and R4 are all-null."""
        result = _pivots(sample_df, method="fibonacci")
        assert len(result.columns) == 9
        s4 = next(c for c in result.columns if c.endswith("_S4"))
        r4 = next(c for c in result.columns if c.endswith("_R4"))
        assert np.isnan(result[s4].to_numpy()).all()
        assert np.isnan(result[r4].to_numpy()).all()

    def test_camarilla_method(self, sample_df):
        """Test Camarilla pivot method."""
        result = _pivots(sample_df, method="camarilla")
        assert len(result.columns) == 9  # P, S1-S4, R1-R4

    def test_classic_method(self, sample_df):
        """Test Classic pivot method."""
        result = _pivots(sample_df, method="classic")
        assert len(result.columns) == 9

    def test_demark_method(self, sample_df):
        """DeMark emits the full struct; only P, S1, R1 are populated."""
        result = _pivots(sample_df, method="demark")
        assert any("_P" in col for col in result.columns)
        s2 = next(c for c in result.columns if c.endswith("_S2"))
        assert np.isnan(result[s2].to_numpy()).all()

    def test_woodie_method(self, sample_df):
        """Test Woodie pivot method."""
        result = _pivots(sample_df, method="woodie")
        assert len(result.columns) == 9

    def test_with_zeros(self, sample_df):
        """Handles zero values."""
        df = sample_df.with_columns([pl.lit(0.0).alias("low").cast(pl.Float64)])
        result = _pivots(df)
        assert isinstance(result, pl.DataFrame)
