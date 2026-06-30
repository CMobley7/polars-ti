# -*- coding: utf-8 -*-
"""Tests for Polars RVGI (Relative Vigor Index) implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.momentum.rvgi import rvgi


class TestPlRvgi:
    """Test suite for pl_rvgi Polars implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLC data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2

        pl_df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
        pd_df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
        return pl_df, pd_df

    def test_returns_expression(self, sample_data):
        """pl_rvgi should return a Polars expression."""
        result = rvgi("open", "high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_structure(self, sample_data):
        """Output should be a struct with RVGI and RVGIs fields."""
        pl_df, _ = sample_data
        result = pl_df.select(rvgi("open", "high", "low", "close", length=14, swma_length=4))

        assert "RVGI" in result.columns
        struct = result["RVGI"]
        assert "RVGI_14_4" in struct.struct.fields
        assert "RVGIs_14_4" in struct.struct.fields

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter should shift results."""
        pl_df, _ = sample_data
        offset = 5

        result_no_offset = pl_df.select(rvgi("open", "high", "low", "close", offset=0))
        result_with_offset = pl_df.select(rvgi("open", "high", "low", "close", offset=offset))

        no_offset_vals = result_no_offset["RVGI"].struct.field("RVGI_14_4").to_numpy()
        with_offset_vals = result_with_offset["RVGI"].struct.field("RVGI_14_4").to_numpy()

        # Find valid index to compare
        valid_idx = 50
        assert np.isclose(no_offset_vals[valid_idx], with_offset_vals[valid_idx + offset], atol=1e-10)

    def test_lazy_execution(self, sample_data):
        """pl_rvgi should work with lazy DataFrames."""
        pl_df, _ = sample_data
        lazy_df = pl_df.lazy()

        result = lazy_df.select(rvgi("open", "high", "low", "close")).collect()
        assert result.height == pl_df.height
        assert "RVGI" in result.columns

    def test_different_parameters(self, sample_data):
        """RVGI should work with different length and swma_length parameters."""
        pl_df, _ = sample_data

        for length in [7, 14, 21]:
            for swma_length in [3, 4, 5]:
                result = pl_df.select(
                    rvgi(
                        "open",
                        "high",
                        "low",
                        "close",
                        length=length,
                        swma_length=swma_length,
                    )
                )
                assert f"RVGI_{length}_{swma_length}" in result["RVGI"].struct.fields
