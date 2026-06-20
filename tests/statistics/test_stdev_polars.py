# -*- coding: utf-8 -*-
"""Tests for pl_stdev."""

import numpy as np
import polars as pl
import pytest
from polars_ti.statistics.stdev import stdev


class TestPlStdev:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {"pd_close": close, "pl_df": pl.DataFrame({"close": close})}

    def test_returns_expression(self):
        assert isinstance(stdev("close"), pl.Expr)

    def test_output_has_correct_alias(self, sample_df):
        assert "STDEV_30" in sample_df.select(stdev("close", length=30)).columns

    def test_with_null_values(self):
        assert pl.DataFrame({"close": [None] + [100.0] * 49}).select(stdev("close")).height == 50

    def test_with_zeros(self):
        assert pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45}).select(stdev("close")).height == 50

    def test_lazy_execution(self, sample_df):
        assert "STDEV_30" in sample_df.lazy().select(stdev("close")).collect().columns
