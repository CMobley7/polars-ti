# -*- coding: utf-8 -*-
"""Tests for pl_tos_stdevall."""

import numpy as np
import polars as pl
import pytest
from polars_ti.statistics.tos_stdevall import tos_stdevall


class TestPlTosStdevall:
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
        assert isinstance(tos_stdevall("close"), pl.Expr)

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(tos_stdevall("close"))
        assert "TOS_STDEVALL" in result.columns

    def test_with_null_values(self):
        assert pl.DataFrame({"close": [None] + [100.0] * 49}).select(tos_stdevall("close")).height == 50

    def test_with_zeros(self):
        assert pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45}).select(tos_stdevall("close")).height == 50

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(tos_stdevall("close")).collect()
        assert "TOS_STDEVALL" in result.columns
