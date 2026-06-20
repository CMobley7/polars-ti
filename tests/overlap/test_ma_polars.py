# -*- coding: utf-8 -*-
"""Tests for pl_ma() dispatcher function."""

import numpy as np
import polars as pl
import pytest
from polars_ti.ma import ma


class TestPlMaDispatcher:
    """Test suite for pl_ma() Polars MA dispatcher."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pl.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})

    def test_returns_available_mas_when_no_args(self):
        """Test that pl_ma returns list of MAs when no args provided."""
        result = ma(name=None, source=None)
        assert isinstance(result, list)
        assert len(result) == 17
        assert "sma" in result
        assert "ema" in result

    def test_all_17_mas_work(self, sample_df):
        """Test that all 17 MA types work correctly."""
        available_mas = ma(name=None, source=None)
        for ma_name in available_mas:
            result = sample_df.select(ma(ma_name, "close", length=10).alias("result"))
            assert result.height == 100
            # All should have some non-null values after warmup
            non_null = result.drop_nulls().height
            assert non_null > 50, f"{ma_name} has too many nulls"

    def test_default_is_ema(self, sample_df):
        """Test that default MA is EMA."""
        result_default = sample_df.select(ma("invalid", "close", length=10).alias("result"))
        result_ema = sample_df.select(ma("ema", "close", length=10).alias("result"))

        arr1 = result_default.get_column("result").to_numpy()
        arr2 = result_ema.get_column("result").to_numpy()

        np.testing.assert_array_equal(arr1, arr2)

    def test_case_insensitive(self, sample_df):
        """Test that MA name is case insensitive."""
        result_upper = sample_df.select(ma("SMA", "close", length=10).alias("result"))
        result_lower = sample_df.select(ma("sma", "close", length=10).alias("result"))

        arr1 = result_upper.get_column("result").to_numpy()
        arr2 = result_lower.get_column("result").to_numpy()

        np.testing.assert_array_equal(arr1, arr2)

    def test_length_parameter(self, sample_df):
        """Test that length parameter works."""
        result_10 = sample_df.select(ma("sma", "close", length=10).alias("result"))
        result_20 = sample_df.select(ma("sma", "close", length=20).alias("result"))

        arr1 = result_10.get_column("result").to_numpy()
        arr2 = result_20.get_column("result").to_numpy()

        # Different lengths should give different results (after warmup)
        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        assert np.any(arr1[mask] != arr2[mask])

    def test_talib_parameter_routing(self, sample_df):
        """Test that talib parameter is correctly routed."""
        result_talib = sample_df.select(ma("sma", "close", length=10, talib=True).alias("result"))
        result_pure = sample_df.select(ma("sma", "close", length=10, talib=False).alias("result"))

        arr1 = result_talib.get_column("result").to_numpy()
        arr2 = result_pure.get_column("result").to_numpy()

        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        max_diff = np.abs(arr1[mask] - arr2[mask]).max()
        assert max_diff < 1e-10, f"SMA talib vs pure diff too high: {max_diff}"

    def test_offset_parameter(self, sample_df):
        """Test that offset parameter works."""
        result_no_offset = sample_df.select(ma("sma", "close", length=10, offset=0).alias("result"))
        result_offset = sample_df.select(ma("sma", "close", length=10, offset=5).alias("result"))

        arr1 = result_no_offset.get_column("result").to_numpy()
        arr2 = result_offset.get_column("result").to_numpy()

        # Offset should shift the results
        assert not np.array_equal(arr1, arr2)

    def test_kwargs_pass_through(self, sample_df):
        """Test that kwargs are passed through correctly."""
        # Test with asc parameter for wma
        result_asc = sample_df.select(ma("wma", "close", length=10, asc=True).alias("result"))
        result_desc = sample_df.select(ma("wma", "close", length=10, asc=False).alias("result"))

        arr1 = result_asc.get_column("result").to_numpy()
        arr2 = result_desc.get_column("result").to_numpy()

        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        assert np.any(arr1[mask] != arr2[mask]), "asc=True and asc=False should differ"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 29})
        result = df.select(ma("sma", "close", length=10))
        assert result.height == 30

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 25})
        result = df.select(ma("sma", "close", length=10))
        assert result.height == 30

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(ma("ema", "close", length=10).alias("result")).collect()
        assert "result" in result.columns
