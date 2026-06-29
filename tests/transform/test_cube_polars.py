# -*- coding: utf-8 -*-
"""Tests for pl_cube."""

import numpy as np
import polars as pl
import pytest
from polars_ti.transform.cube import cube


class TestPlCube:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = np.random.randn(100) * 0.5
        return close

    def test_returns_expressions(self, sample_data):
        result = cube("close")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.select(cube("close", pwr=3.0, signal_offset=-1))
        assert "CUBE_3.0_-1" in result.columns
        assert "CUBEs_3.0_-1" in result.columns

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data[10:15] = np.nan
        df = pl.DataFrame({"close": data})
        result = df.select(cube("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data[20:25] = 0.0
        df = pl.DataFrame({"close": data})
        result = df.select(cube("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.lazy().select(cube("close")).collect()
        assert result.height == 100
