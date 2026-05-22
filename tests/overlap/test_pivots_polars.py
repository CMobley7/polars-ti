# -*- coding: utf-8 -*-
"""Tests for pl_pivots."""

import datetime
import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.pivots import pl_pivots


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

    def test_returns_dataframe(self, sample_df):
        result = pl_pivots(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_pivot_columns(self, sample_df):
        result = pl_pivots(sample_df)
        assert any("_P" in col for col in result.columns)
        assert any("_S1" in col for col in result.columns)
        assert any("_R1" in col for col in result.columns)

    def test_traditional_method(self, sample_df):
        result = pl_pivots(sample_df, method="traditional")
        assert len(result.columns) == 9  # P, S1-S4, R1-R4

    def test_fibonacci_method(self, sample_df):
        result = pl_pivots(sample_df, method="fibonacci")
        assert len(result.columns) == 7  # P, S1-S3, R1-R3

    def test_camarilla_method(self, sample_df):
        """Test Camarilla pivot method."""
        result = pl_pivots(sample_df, method="camarilla")
        assert len(result.columns) == 9  # P, S1-S4, R1-R4

    def test_classic_method(self, sample_df):
        """Test Classic pivot method."""
        result = pl_pivots(sample_df, method="classic")
        assert len(result.columns) == 9

    def test_demark_method(self, sample_df):
        """Test DeMark pivot method."""
        result = pl_pivots(sample_df, method="demark")
        assert any("_P" in col for col in result.columns)

    def test_woodie_method(self, sample_df):
        """Test Woodie pivot method."""
        result = pl_pivots(sample_df, method="woodie")
        assert len(result.columns) == 9

    def test_with_zeros(self, sample_df):
        """Handles zero values."""
        df = sample_df.with_columns([pl.lit(0.0).alias("low").cast(pl.Float64)])
        result = pl_pivots(df)
        assert isinstance(result, pl.DataFrame)
