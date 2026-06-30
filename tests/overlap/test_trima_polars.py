# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/trima.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.trima import trima


class TestPlTrima:
    """Tests for pl_trima - Triangular Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            "pd_series": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_expression(self):
        """Returns a Polars expression."""
        result = trima("close", length=10)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data["pl_df"].select(trima("close", length=10))
        assert result.columns[0] == "TRIMA_10"

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter shifts the result."""
        no_offset = sample_data["pl_df"].select(trima("close", length=10)).to_series()
        with_offset = sample_data["pl_df"].select(trima("close", length=10, offset=5)).to_series()

        for i in range(15, 40):
            if not np.isnan(no_offset[i]):
                assert no_offset[i] == with_offset[i + 5], f"Offset mismatch at {i}"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(trima("close", length=10))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(trima("close", length=10))
        assert result.height == 30

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data["pl_df"].lazy()
        result = lazy_df.select(trima("close", length=10)).collect()
        assert "TRIMA_10" in result.columns

    @pytest.mark.parametrize("length", [9, 10, 14, 20])
    def test_native_matches_talib_reference(self, length):
        """Native TRIMA (ceil/floor asymmetric windows, classic 41c91db) must
        match talib.TRIMA. The old symmetric round(0.5*(length+1)) window
        diverged for even lengths (off by ~0.57 on SPY for length=10)."""
        talib = pytest.importorskip("talib")
        df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
        close = df["close"].to_numpy().astype(float)
        ref = talib.TRIMA(close, timeperiod=length)
        got = df.select(trima("close", length=length, talib=False)).to_series().to_numpy()
        mask = ~np.isnan(ref) & ~np.isnan(got)
        assert mask.sum() > 100
        assert np.max(np.abs(ref[mask] - got[mask])) < 1e-9
