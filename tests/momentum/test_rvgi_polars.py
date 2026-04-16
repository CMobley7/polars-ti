# -*- coding: utf-8 -*-
"""Tests for Polars RVGI (Relative Vigor Index) implementation."""
import numpy as np
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
import polars as pl
import pytest

from polars_ti.momentum.rvgi import pl_rvgi


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
        pd_df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
        return pl_df, pd_df

    def test_returns_expression(self, sample_data):
        """pl_rvgi should return a Polars expression."""
        result = pl_rvgi("open", "high", "low", "close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_structure(self, sample_data):
        """Output should be a struct with RVGI and RVGIs fields."""
        pl_df, _ = sample_data
        result = pl_df.select(pl_rvgi("open", "high", "low", "close", length=14, swma_length=4))
        
        assert "RVGI" in result.columns
        struct = result["RVGI"]
        assert "RVGI_14_4" in struct.struct.fields
        assert "RVGIs_14_4" in struct.struct.fields

    def test_numerical_parity_with_pandas(self, sample_data):
        """Polars RVGI should match Pandas RVGI within 1e-6 tolerance."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        pl_df, pd_df = sample_data
        length = 14
        swma_length = 4

        # Pandas result
        pd_result = rvgi(pd_df["open"], pd_df["high"], pd_df["low"], pd_df["close"],
                        length=length, swma_length=swma_length)
        pd_rvgi = pd_result[f"RVGI_{length}_{swma_length}"].to_numpy()
        pd_signal = pd_result[f"RVGIs_{length}_{swma_length}"].to_numpy()

        # Polars result
        pl_result = pl_df.select(pl_rvgi("open", "high", "low", "close", length=length, swma_length=swma_length))
        pl_struct = pl_result["RVGI"]
        pl_rvgi_arr = pl_struct.struct.field(f"RVGI_{length}_{swma_length}").to_numpy()
        pl_signal_arr = pl_struct.struct.field(f"RVGIs_{length}_{swma_length}").to_numpy()

        # Compare after warmup
        warmup = length + swma_length + 10
        mask_rvgi = ~np.isnan(pd_rvgi[warmup:]) & ~np.isnan(pl_rvgi_arr[warmup:])
        mask_signal = ~np.isnan(pd_signal[warmup:]) & ~np.isnan(pl_signal_arr[warmup:])

        rvgi_max_diff = np.max(np.abs(pl_rvgi_arr[warmup:][mask_rvgi] - pd_rvgi[warmup:][mask_rvgi]))
        signal_max_diff = np.max(np.abs(pl_signal_arr[warmup:][mask_signal] - pd_signal[warmup:][mask_signal]))

        assert rvgi_max_diff < 1e-6, f"RVGI max diff {rvgi_max_diff} exceeds tolerance"
        assert signal_max_diff < 1e-6, f"Signal max diff {signal_max_diff} exceeds tolerance"

    def test_offset_shifts_result(self, sample_data):
        """Offset parameter should shift results."""
        pl_df, _ = sample_data
        offset = 5

        result_no_offset = pl_df.select(pl_rvgi("open", "high", "low", "close", offset=0))
        result_with_offset = pl_df.select(pl_rvgi("open", "high", "low", "close", offset=offset))

        no_offset_vals = result_no_offset["RVGI"].struct.field("RVGI_14_4").to_numpy()
        with_offset_vals = result_with_offset["RVGI"].struct.field("RVGI_14_4").to_numpy()

        # Find valid index to compare
        valid_idx = 50
        assert np.isclose(
            no_offset_vals[valid_idx],
            with_offset_vals[valid_idx + offset],
            atol=1e-10
        )

    def test_lazy_execution(self, sample_data):
        """pl_rvgi should work with lazy DataFrames."""
        pl_df, _ = sample_data
        lazy_df = pl_df.lazy()

        result = lazy_df.select(pl_rvgi("open", "high", "low", "close")).collect()
        assert result.height == pl_df.height
        assert "RVGI" in result.columns

    def test_different_parameters(self, sample_data):
        """RVGI should work with different length and swma_length parameters."""
        pl_df, _ = sample_data
        
        for length in [7, 14, 21]:
            for swma_length in [3, 4, 5]:
                result = pl_df.select(pl_rvgi("open", "high", "low", "close", 
                                              length=length, swma_length=swma_length))
                assert f"RVGI_{length}_{swma_length}" in result["RVGI"].struct.fields
