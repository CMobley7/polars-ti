# -*- coding: utf-8 -*-
"""Tests for pl_exhc - Polars + Numba implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.exhc import pl_exhc


class TestPlExhc:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_list_of_expr(self):
        result = pl_exhc("close")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_has_exhc_columns(self, sample_df):
        result = sample_df.select(pl_exhc("close"))
        assert "EXHC_DNa" in result.columns
        assert "EXHC_UPa" in result.columns

    def test_show_all_false(self, sample_df):
        result = sample_df.select(pl_exhc("close", show_all=False))
        assert "EXHC_DN" in result.columns
        assert "EXHC_UP" in result.columns

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_exhc("close", offset=0))
        result_with_offset = sample_df.select(pl_exhc("close", offset=5))
        assert result_with_offset["EXHC_DNa"].null_count() > result_no_offset["EXHC_DNa"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_exhc("close")).collect()
        assert "EXHC_DNa" in result.columns

    def test_asint_parameter(self, sample_df):
        result = sample_df.select(pl_exhc("close", asint=True))
        assert result["EXHC_DNa"].dtype == pl.Int64

    def test_values_non_negative(self, sample_df):
        result = sample_df.select(pl_exhc("close"))
        valid_dn = result["EXHC_DNa"].filter(~result["EXHC_DNa"].is_null())
        valid_up = result["EXHC_UPa"].filter(~result["EXHC_UPa"].is_null())
        assert valid_dn.min() >= 0
        assert valid_up.min() >= 0
