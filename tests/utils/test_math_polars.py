# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/utils/_math.py Polars utilities."""
import numpy as np
import polars as pl
import pytest

from polars_ti.utils._math import (
    pl_fibonacci_weights,
    pl_non_zero_range,
    pl_pascals_triangle_weights,
    pl_signed_diff,
    pl_symmetric_triangle_weights,
    pl_unsigned_differences,
)


class TestPlNonZeroRange:
    """Tests for pl_non_zero_range."""

    def test_basic_range(self):
        """Test basic high-low range calculation."""
        df = pl.DataFrame({"high": [10.0, 11.0, 12.0], "low": [9.0, 10.0, 11.0]})
        result = df.select(pl_non_zero_range("high", "low").alias("range"))
        expected = [1.0, 1.0, 1.0]
        assert result["range"].to_list() == expected

    def test_zero_range_gets_epsilon(self):
        """Test that zero range gets epsilon added."""
        df = pl.DataFrame({"high": [10.0, 10.0, 10.0], "low": [10.0, 9.0, 10.0]})
        result = df.select(pl_non_zero_range("high", "low").alias("range"))
        # First and third should have epsilon, second should be 1.0
        assert result["range"][0] > 0  # epsilon
        assert result["range"][1] == 1.0
        assert result["range"][2] > 0  # epsilon
        assert result["range"][0] < 1e-10  # Very small (epsilon)


class TestPlSignedDiff:
    """Tests for pl_signed_diff."""

    def test_increasing_values(self):
        """Test increasing values return 1."""
        df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(pl_signed_diff("close").alias("sign"))
        # First is null (no diff), rest should be 1
        assert result["sign"].to_list()[1:] == [1, 1, 1, 1]

    def test_decreasing_values(self):
        """Test decreasing values return -1."""
        df = pl.DataFrame({"close": [5.0, 4.0, 3.0, 2.0, 1.0]})
        result = df.select(pl_signed_diff("close").alias("sign"))
        assert result["sign"].to_list()[1:] == [-1, -1, -1, -1]

    def test_unchanged_values(self):
        """Test unchanged values return 0."""
        df = pl.DataFrame({"close": [5.0, 5.0, 5.0]})
        result = df.select(pl_signed_diff("close").alias("sign"))
        assert result["sign"].to_list()[1:] == [0, 0]


class TestPlUnsignedDifferences:
    """Tests for pl_unsigned_differences."""

    def test_positive_negative_separation(self):
        """Test that positive and negative are separated correctly."""
        df = pl.DataFrame({"close": [1.0, 2.0, 1.5, 3.0, 2.5]})
        pos, neg = pl_unsigned_differences("close")
        result = df.select(pos.alias("pos"), neg.alias("neg"))
        
        # Diff: [null, +1, -0.5, +1.5, -0.5]
        # Positive: [0, 1, 0, 1, 0]
        # Negative: [0, 0, 1, 0, 1]
        assert result["pos"].to_list() == [0, 1, 0, 1, 0]
        assert result["neg"].to_list() == [0, 0, 1, 0, 1]


class TestWeightGenerators:
    """Tests for weight generator functions."""

    def test_fibonacci_weights_length(self):
        """Test fibonacci weights returns correct length."""
        weights = pl_fibonacci_weights(5)
        assert len(weights) == 5

    def test_fibonacci_weights_sum_to_one(self):
        """Test fibonacci weights sum to approximately 1."""
        weights = pl_fibonacci_weights(10)
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_symmetric_triangle_weights_length(self):
        """Test symmetric triangle weights returns correct length."""
        weights = pl_symmetric_triangle_weights(5)
        assert len(weights) == 5

    def test_symmetric_triangle_weights_sum_to_one(self):
        """Test symmetric triangle weights sum to approximately 1."""
        weights = pl_symmetric_triangle_weights(10)
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_pascals_triangle_weights_length(self):
        """Test Pascal's triangle weights returns correct length."""
        weights = pl_pascals_triangle_weights(5)
        assert len(weights) == 6  # n+1 elements

    def test_pascals_triangle_weights_sum_to_one(self):
        """Test Pascal's triangle weights sum to approximately 1."""
        weights = pl_pascals_triangle_weights(10)
        assert abs(sum(weights) - 1.0) < 1e-10
