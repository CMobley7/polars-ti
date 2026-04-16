# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/utils/_time.py Polars utilities."""
import polars as pl
import pytest
from datetime import datetime, timedelta

from polars_ti.utils._time import (
    pl_total_time,
    pl_filter_dates,
    pl_year_to_date,
    pl_month_to_date,
)


class TestPlTotalTime:
    """Tests for pl_total_time."""

    def test_years_calculation(self):
        """Test years calculation."""
        # Create ~2 years of data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(730)]
        df = pl.DataFrame({"date": dates, "close": list(range(730))})
        result = pl_total_time(df, "date", tf="years")
        assert 1.99 < result < 2.01

    def test_days_calculation(self):
        """Test days calculation."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame({"date": dates, "close": list(range(100))})
        result = pl_total_time(df, "date", tf="days")
        assert 98 < result < 100

    def test_empty_dataframe(self):
        """Test empty DataFrame returns 0."""
        df = pl.DataFrame({"date": [], "close": []}).cast({"date": pl.Datetime})
        result = pl_total_time(df, "date", tf="years")
        assert result == 0.0


class TestPlFilterDates:
    """Tests for pl_filter_dates."""

    def test_filter_specific_dates(self):
        """Test filtering to specific dates."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        df = pl.DataFrame({"date": dates, "value": [1, 2, 3]})
        result = pl_filter_dates(df, "date", ["2024-01-01", "2024-01-03"])
        assert len(result) == 2
        assert result["value"].to_list() == [1, 3]


class TestPlYearToDate:
    """Tests for pl_year_to_date."""

    def test_filters_to_current_year(self):
        """Test that records before current year are excluded."""
        now = datetime.now()
        last_year = now.replace(year=now.year - 1)
        dates = [last_year, now]
        df = pl.DataFrame({"date": dates, "value": [1, 2]})
        result = pl_year_to_date(df, "date")
        # Should only include current year
        assert len(result) == 1
        assert result["value"][0] == 2


class TestPlMonthToDate:
    """Tests for pl_month_to_date."""

    def test_filters_to_current_month(self):
        """Test that records before current month are excluded."""
        now = datetime.now()
        if now.month > 1:
            last_month = now.replace(month=now.month - 1)
        else:
            last_month = now.replace(year=now.year - 1, month=12)
        dates = [last_month, now]
        df = pl.DataFrame({"date": dates, "value": [1, 2]})
        result = pl_month_to_date(df, "date")
        # Should only include current month
        assert len(result) == 1
        assert result["value"][0] == 2
