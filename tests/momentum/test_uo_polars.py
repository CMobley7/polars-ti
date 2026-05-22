# -*- coding: utf-8 -*-
"""Tests for Polars UO (Ultimate Oscillator) implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.uo import pl_uo


class TestPlUo:
    """Test suite for pl_uo function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_basic_output(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_uo returns correct column."""
        result = sample_data.select(pl_uo("high", "low", "close", talib=False))

        assert result.shape[1] == 1
        assert "UO_7_14_28" in result.columns

    def test_custom_parameters(self, sample_data: pl.DataFrame) -> None:
        """Test pl_uo with custom periods."""
        result = sample_data.select(pl_uo("high", "low", "close", fast=5, medium=10, slow=20, talib=False))

        assert "UO_5_10_20" in result.columns

    def test_uo_range(self, sample_data: pl.DataFrame) -> None:
        """Test that UO values are within 0 to 100."""
        result = sample_data.select(pl_uo("high", "low", "close", talib=False))

        uo = result["UO_7_14_28"].to_numpy()
        valid = ~np.isnan(uo)

        # UO should be between 0 and 100
        assert np.all(uo[valid] >= 0)
        assert np.all(uo[valid] <= 100)

    def test_offset_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test offset shifts output."""
        result_no_offset = sample_data.select(pl_uo("high", "low", "close", offset=0, talib=False))
        result_with_offset = sample_data.select(pl_uo("high", "low", "close", offset=5, talib=False))

        uo_no = result_no_offset["UO_7_14_28"].to_numpy()
        uo_with = result_with_offset["UO_7_14_28"].to_numpy()

        # Offset version should have more leading NaNs
        mask_no = ~np.isnan(uo_no)
        mask_with = ~np.isnan(uo_with)

        assert np.sum(mask_no) > np.sum(mask_with)

    def test_custom_weights(self, sample_data: pl.DataFrame) -> None:
        """Test that custom weights affect output."""
        result_default = sample_data.select(
            pl_uo(
                "high",
                "low",
                "close",
                fast_w=4.0,
                medium_w=2.0,
                slow_w=1.0,
                talib=False,
            )
        )
        result_custom = sample_data.select(
            pl_uo(
                "high",
                "low",
                "close",
                fast_w=1.0,
                medium_w=1.0,
                slow_w=1.0,
                talib=False,
            )
        )

        uo_default = result_default["UO_7_14_28"].to_numpy()
        uo_custom = result_custom["UO_7_14_28"].to_numpy()

        # Different weights should give different results
        mask = ~np.isnan(uo_default) & ~np.isnan(uo_custom)
        assert not np.allclose(uo_default[mask][:50], uo_custom[mask][:50], rtol=1e-6)

    def test_lazy_evaluation(self, sample_data: pl.DataFrame) -> None:
        """Test that pl_uo works in lazy context."""
        lazy_df = sample_data.lazy()
        result = lazy_df.select(pl_uo("high", "low", "close", talib=False)).collect()

        assert result.shape[1] == 1
        assert not result["UO_7_14_28"].is_empty()

    def test_null_handling(self) -> None:
        """Test pl_uo handles null values gracefully."""
        n = 200
        close = [100.0 + i * 0.1 for i in range(n)]
        high = [c + 0.5 for c in close]
        low = [c - 0.5 for c in close]

        df = pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
            }
        )

        result = df.select(pl_uo("high", "low", "close", talib=False))

        assert result.shape[0] == n
        uo = result["UO_7_14_28"].to_numpy()
        assert np.isnan(uo[0])  # First value should be NaN

    def test_expr_input(self, sample_data: pl.DataFrame) -> None:
        """Test pl_uo accepts pl.Expr as input."""
        result = sample_data.select(pl_uo(pl.col("high"), pl.col("low"), pl.col("close"), talib=False))

        assert result.shape[1] == 1

    def test_talib_parameter(self, sample_data: pl.DataFrame) -> None:
        """Test that talib parameter controls TA-Lib usage."""
        # Both should work regardless of TA-Lib availability
        result_talib = sample_data.select(pl_uo("high", "low", "close", talib=True))
        result_pure = sample_data.select(pl_uo("high", "low", "close", talib=False))

        assert result_talib.shape[1] == 1
        assert result_pure.shape[1] == 1

    def test_talib_parity(self, sample_data: pl.DataFrame) -> None:
        """Test parity between talib=True and talib.ULTOSC directly."""
        try:
            from talib import ULTOSC

            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False
            pytest.skip("TA-Lib not available")

        if HAS_TALIB:
            h = sample_data["high"].to_numpy()
            l = sample_data["low"].to_numpy()
            c = sample_data["close"].to_numpy()

            talib_result = ULTOSC(h, l, c, 7, 14, 28)
            polars_result = sample_data.select(pl_uo("high", "low", "close", talib=True))
            polars_val = polars_result["UO_7_14_28"].to_numpy()

            mask = ~np.isnan(talib_result) & ~np.isnan(polars_val)
            np.testing.assert_allclose(talib_result[mask], polars_val[mask], rtol=1e-10, atol=1e-14)
