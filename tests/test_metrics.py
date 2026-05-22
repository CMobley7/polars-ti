# -*- coding: utf-8 -*-
from pytest import mark

import polars_ti as ti


def test_cagr(df):
    result = ti.utils.cagr(df.close)
    assert isinstance(result, float)


def test_calmar_ratio(df):
    result = ti.utils.calmar_ratio(df.close)
    assert isinstance(result, float)


@mark.parametrize("year,result", [(-2, None), (0, None)])
def test_calmar_ratio_year(df, year, result):
    assert ti.utils.calmar_ratio(df.close, years=year) is result


def test_downside_deviation(df):
    pctret = df.ti.percent_return()
    result = ti.utils.downside_deviation(pctret)
    assert isinstance(result, float)

    logret = df.ti.percent_return()
    result = ti.utils.downside_deviation(logret)
    assert isinstance(result, float)


def test_jensens_alpha(df):
    bench_return = df.ti.percent_return().sample(n=df.close.shape[0], random_state=1)
    result = ti.utils.jensens_alpha(df.close, bench_return)
    assert isinstance(result, float)


def test_log_max_drawdown(df):
    result = ti.utils.log_max_drawdown(df.close)
    assert isinstance(result, float)


def test_max_drawdown(df):
    result = ti.utils.max_drawdown(df.close)
    assert isinstance(result, float)

    result = ti.utils.max_drawdown(df.close, method="percent")
    assert isinstance(result, float)

    result = ti.utils.max_drawdown(df.close, method="log")
    assert isinstance(result, float)

    result = ti.utils.max_drawdown(df.close, all=True)
    assert isinstance(result, dict)
    assert isinstance(result["dollar"], float)
    assert isinstance(result["percent"], float)
    assert isinstance(result["log"], float)


def test_optimal_leverage(df):
    result = ti.utils.optimal_leverage(df.close)
    assert isinstance(result, int)

    result = ti.utils.optimal_leverage(df.close, log=True)
    assert isinstance(result, int)


def test_pure_profit_score(df):
    result = ti.utils.pure_profit_score(df.close)
    assert isinstance(result, float)


def test_sharpe_ratio(df):
    result = ti.utils.sharpe_ratio(df.close)
    assert isinstance(result, float)


def test_sortino_ratio(df):
    result = ti.utils.sortino_ratio(df.close)
    assert isinstance(result, float)


def test_volatility(df):
    pctret = df.ti.percent_return()
    result = ti.utils.volatility(pctret, returns=True)
    assert isinstance(result, float)


@mark.parametrize("tf", ["years", "months", "weeks", "days", "hours", "minutes", "seconds"])
def test_volatility_timeframe(df, tf):
    assert isinstance(ti.utils.volatility(df.close, tf), float)
