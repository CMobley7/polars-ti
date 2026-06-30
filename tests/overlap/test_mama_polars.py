# -*- coding: utf-8 -*-
"""Tests for mama."""

import numpy as np
import polars as pl
import pytest
from polars_ti.overlap.mama import mama


def _mama(df, **kwargs):
    """Helper: run mama and return the unnested struct as a flat DataFrame."""
    result = df.select(mama("close", **kwargs))
    return result.unnest(result.columns[0])


class TestPlMama:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            }
        )

    def test_returns_expression(self, sample_df):
        assert isinstance(mama("close"), pl.Expr)

    def test_columns_present(self, sample_df):
        result = _mama(sample_df)
        assert "MAMA_0.5_0.05" in result.columns
        assert "FAMA_0.5_0.05" in result.columns

    def test_talib_version(self, sample_df):
        result = _mama(sample_df, talib=True)
        assert "MAMA_0.5_0.05" in result.columns

    def test_pure_version(self, sample_df):
        result = _mama(sample_df, talib=False)
        assert "MAMA_0.5_0.05" in result.columns

    def test_custom_limits(self, sample_df):
        result = _mama(sample_df, fastlimit=0.4, slowlimit=0.1)
        assert "MAMA_0.4_0.1" in result.columns
        assert "FAMA_0.4_0.1" in result.columns
