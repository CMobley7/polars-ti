# -*- coding: utf-8 -*-
"""Tests for pl_cube."""
import numpy as np
import polars as pl
import pytest
from polars_ti.transform.cube import pl_cube


class TestPlCube:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = np.random.randn(100) * 0.5
        return close

    def test_returns_expression(self, sample_data):
        result = pl_cube("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.select(pl_cube("close", pwr=3.0, signal_offset=-1))
        assert "CUBE_3.0_-1" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data[10:15] = np.nan
        df = pl.DataFrame({"close": data})
        result = df.select(pl_cube("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data[20:25] = 0.0
        df = pl.DataFrame({"close": data})
        result = df.select(pl_cube("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.lazy().select(pl_cube("close")).collect()
        assert result.height == 100
