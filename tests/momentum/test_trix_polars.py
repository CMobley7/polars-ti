# -*- coding: utf-8 -*-
"""Tests for pl_trix - Polars + Numba TRIX with TA-Lib support."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.trix import trix as trix_indicator


class TestPlTrix:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 150
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expr(self):
        expr = trix_indicator("close")
        assert isinstance(expr, pl.Expr)

    def test_has_trix_column(self, sample_df):
        result = sample_df.select(trix_indicator("close"))
        assert "TRIX" in result.columns

    def test_struct_has_all_fields(self, sample_df):
        result = sample_df.select(trix_indicator("close"))
        trix = result["TRIX"]
        assert "TRIX_30_9" in trix.struct.fields
        assert "TRIXs_30_9" in trix.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(trix_indicator("close"))
        trix = result["TRIX"]
        main = trix.struct.field("TRIX_30_9")
        # After warmup period, values should be valid
        assert main[100:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(trix_indicator("close", offset=5))
        trix = result["TRIX"]
        main = trix.struct.field("TRIX_30_9")
        assert main[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(trix_indicator("close")).collect()
        assert "TRIX" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(trix_indicator("close", talib=True))
        assert "TRIX" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(trix_indicator("close", talib=False))
        assert "TRIX" in result.columns

    def test_custom_length_parameter(self, sample_df):
        result = sample_df.select(trix_indicator("close", length=20, signal=5))
        trix = result["TRIX"]
        assert "TRIX_20_5" in trix.struct.fields

    def test_length_signal_swap(self, sample_df):
        """If length < signal, they should be swapped."""
        result = sample_df.select(trix_indicator("close", length=5, signal=20))
        trix = result["TRIX"]
        assert "TRIX_20_5" in trix.struct.fields

    def test_with_null_values(self):
        df = pl.DataFrame({"close": [None] + [100.0] * 149})
        result = df.select(trix_indicator("close"))
        assert result.height == 150

    def test_talib_parity(self):
        """Verify TA-Lib path produces same results as direct TA-Lib call."""
        try:
            from talib import TRIX as TALIB_TRIX
        except ImportError:
            pytest.skip("TA-Lib not installed")

        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        pldf = pl.DataFrame({"close": close})

        talib_trix = TALIB_TRIX(close, timeperiod=30)

        polars_result = pldf.select(trix_indicator("close", talib=True))
        trix = polars_result["TRIX"]
        polars_main = trix.struct.field("TRIX_30_9").to_numpy()

        valid_mask = ~np.isnan(talib_trix) & ~np.isnan(polars_main)
        max_diff = np.max(np.abs(talib_trix[valid_mask] - polars_main[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
