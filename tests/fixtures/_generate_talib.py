# -*- coding: utf-8 -*-
"""Generate the direct TA-Lib ground-truth reference fixture.

For every indicator that has a 1:1 TA-Lib function, compute TA-Lib directly on
the SAME deterministic SPY_D slice used by ``_generate_old.py`` and write the
result to ``tests/fixtures/talib_reference.parquet``.

This is the canonical oracle for the "new is better" parity exceptions
(WCP/MFI/ADX/RSI/TRIX and the candle patterns where the OLD library was wrong)
and for the generic TA-Lib-parity checks. Column names are chosen to match the
library's output names where reasonable so the parity engine can align them.

Runs in the NEW venv (it only needs talib + polars), but is environment-neutral
about which ``polars_ti`` is importable because it does NOT import polars_ti.

EXACT COMMAND TO REGENERATE:

    /home/cmobley/Documents/Projects/polars-ti/.venv/bin/python \
        /home/cmobley/Documents/Projects/polars-ti/tests/fixtures/_generate_talib.py

Output:
    tests/fixtures/talib_reference.parquet
"""

import polars as pl
import talib

ROOT = "/home/cmobley/Documents/Projects/polars-ti"
DATA = f"{ROOT}/data/SPY_D.csv"
OUT = f"{ROOT}/tests/fixtures"

# Same deterministic slice as _generate_old.py so rows align by date.
SLICE_ROWS = 1500

raw = pl.read_csv(DATA, try_parse_dates=True).head(SLICE_ROWS)
date = raw["date"]
O = raw["open"].to_numpy().astype("float64")
H = raw["high"].to_numpy().astype("float64")
L = raw["low"].to_numpy().astype("float64")
C = raw["close"].to_numpy().astype("float64")
V = raw["volume"].to_numpy().astype("float64")

cols: dict[str, object] = {"date": date}

# ---- Overlap / moving averages (library default length 10 where applicable) ----
cols["SMA_10"] = talib.SMA(C, 10)
cols["EMA_10"] = talib.EMA(C, 10)
cols["WMA_10"] = talib.WMA(C, 10)
cols["DEMA_10"] = talib.DEMA(C, 10)
cols["TEMA_10"] = talib.TEMA(C, 10)
cols["TRIMA_10"] = talib.TRIMA(C, 10)
cols["MIDPOINT_2"] = talib.MIDPOINT(C, 2)
cols["MIDPRICE_2"] = talib.MIDPRICE(H, L, 2)
cols["WCP"] = talib.WCLPRICE(H, L, C)

# ---- Momentum ----
cols["RSI_14"] = talib.RSI(C, 14)
cols["MFI_14"] = talib.MFI(H, L, C, V, 14)
cols["ADX_14"] = talib.ADX(H, L, C, 14)
cols["DMP_14"] = talib.PLUS_DI(H, L, C, 14)
cols["DMN_14"] = talib.MINUS_DI(H, L, C, 14)
cols["CCI_14_0.015"] = talib.CCI(H, L, C, 14)
cols["MOM_10"] = talib.MOM(C, 10)
cols["ROC_10"] = talib.ROC(C, 10)
cols["WILLR_14"] = talib.WILLR(H, L, C, 14)
cols["CMO_14"] = talib.CMO(C, 14)
cols["TRIX_30_9"] = talib.TRIX(C, 30)

# MACD (12, 26, 9)
_macd, _macds, _macdh = talib.MACD(C, 12, 26, 9)
cols["MACD_12_26_9"] = _macd
cols["MACDh_12_26_9"] = _macdh
cols["MACDs_12_26_9"] = _macds

# ---- Bollinger Bands (5, 2.0) ----
_bbu, _bbm, _bbl = talib.BBANDS(C, 5, 2.0, 2.0)
cols["BBL_5_2.0"] = _bbl
cols["BBM_5_2.0"] = _bbm
cols["BBU_5_2.0"] = _bbu

# ---- Volatility ----
cols["ATRr_14"] = talib.ATR(H, L, C, 14)
cols["NATR_14"] = talib.NATR(H, L, C, 14)
cols["TRUERANGE_1"] = talib.TRANGE(H, L, C)

# ---- Volume ----
cols["OBV"] = talib.OBV(C, V)
cols["AD"] = talib.AD(H, L, C, V)
cols["ADOSC_3_10"] = talib.ADOSC(H, L, C, V, 3, 10)

# ---- Stochastics ----
_sk, _sd = talib.STOCH(H, L, C, 14, 3, 0, 3, 0)
cols["STOCHk_14_3_3"] = _sk
cols["STOCHd_14_3_3"] = _sd
_sfk, _sfd = talib.STOCHF(H, L, C, 14, 3, 0)
cols["STOCHFk_14_3"] = _sfk
cols["STOCHFd_14_3"] = _sfd

# ---- Linear regression family ----
cols["LINEARREG_14"] = talib.LINEARREG(C, 14)

# ---- Candlestick patterns (all 62 the library emits, including the 3 the OLD
# native library got wrong: HIGHWAVE, RICKSHAWMAN, SPINNINGTOP). ----
CDL = {
    "CDL_2CROWS": talib.CDL2CROWS,
    "CDL_3BLACKCROWS": talib.CDL3BLACKCROWS,
    "CDL_3INSIDE": talib.CDL3INSIDE,
    "CDL_3LINESTRIKE": talib.CDL3LINESTRIKE,
    "CDL_3OUTSIDE": talib.CDL3OUTSIDE,
    "CDL_3STARSINSOUTH": talib.CDL3STARSINSOUTH,
    "CDL_3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDL_ABANDONEDBABY": talib.CDLABANDONEDBABY,
    "CDL_ADVANCEBLOCK": talib.CDLADVANCEBLOCK,
    "CDL_BELTHOLD": talib.CDLBELTHOLD,
    "CDL_BREAKAWAY": talib.CDLBREAKAWAY,
    "CDL_CLOSINGMARUBOZU": talib.CDLCLOSINGMARUBOZU,
    "CDL_CONCEALBABYSWALL": talib.CDLCONCEALBABYSWALL,
    "CDL_COUNTERATTACK": talib.CDLCOUNTERATTACK,
    "CDL_DARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDL_DOJISTAR": talib.CDLDOJISTAR,
    "CDL_DRAGONFLYDOJI": talib.CDLDRAGONFLYDOJI,
    "CDL_ENGULFING": talib.CDLENGULFING,
    "CDL_EVENINGDOJISTAR": talib.CDLEVENINGDOJISTAR,
    "CDL_EVENINGSTAR": talib.CDLEVENINGSTAR,
    "CDL_GAPSIDESIDEWHITE": talib.CDLGAPSIDESIDEWHITE,
    "CDL_GRAVESTONEDOJI": talib.CDLGRAVESTONEDOJI,
    "CDL_HAMMER": talib.CDLHAMMER,
    "CDL_HANGINGMAN": talib.CDLHANGINGMAN,
    "CDL_HARAMI": talib.CDLHARAMI,
    "CDL_HARAMICROSS": talib.CDLHARAMICROSS,
    "CDL_HIGHWAVE": talib.CDLHIGHWAVE,
    "CDL_HIKKAKE": talib.CDLHIKKAKE,
    "CDL_HIKKAKEMOD": talib.CDLHIKKAKEMOD,
    "CDL_HOMINGPIGEON": talib.CDLHOMINGPIGEON,
    "CDL_IDENTICAL3CROWS": talib.CDLIDENTICAL3CROWS,
    "CDL_INNECK": talib.CDLINNECK,
    "CDL_INVERTEDHAMMER": talib.CDLINVERTEDHAMMER,
    "CDL_KICKING": talib.CDLKICKING,
    "CDL_KICKINGBYLENGTH": talib.CDLKICKINGBYLENGTH,
    "CDL_LADDERBOTTOM": talib.CDLLADDERBOTTOM,
    "CDL_LONGLEGGEDDOJI": talib.CDLLONGLEGGEDDOJI,
    "CDL_LONGLINE": talib.CDLLONGLINE,
    "CDL_MARUBOZU": talib.CDLMARUBOZU,
    "CDL_MATCHINGLOW": talib.CDLMATCHINGLOW,
    "CDL_MATHOLD": talib.CDLMATHOLD,
    "CDL_MORNINGDOJISTAR": talib.CDLMORNINGDOJISTAR,
    "CDL_MORNINGSTAR": talib.CDLMORNINGSTAR,
    "CDL_ONNECK": talib.CDLONNECK,
    "CDL_PIERCING": talib.CDLPIERCING,
    "CDL_RICKSHAWMAN": talib.CDLRICKSHAWMAN,
    "CDL_RISEFALL3METHODS": talib.CDLRISEFALL3METHODS,
    "CDL_SEPARATINGLINES": talib.CDLSEPARATINGLINES,
    "CDL_SHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
    "CDL_SHORTLINE": talib.CDLSHORTLINE,
    "CDL_SPINNINGTOP": talib.CDLSPINNINGTOP,
    "CDL_STALLEDPATTERN": talib.CDLSTALLEDPATTERN,
    "CDL_STICKSANDWICH": talib.CDLSTICKSANDWICH,
    "CDL_TAKURI": talib.CDLTAKURI,
    "CDL_TASUKIGAP": talib.CDLTASUKIGAP,
    "CDL_THRUSTING": talib.CDLTHRUSTING,
    "CDL_TRISTAR": talib.CDLTRISTAR,
    "CDL_UNIQUE3RIVER": talib.CDLUNIQUE3RIVER,
    "CDL_UPSIDEGAP2CROWS": talib.CDLUPSIDEGAP2CROWS,
    "CDL_XSIDEGAP3METHODS": talib.CDLXSIDEGAP3METHODS,
}
for name, fn in CDL.items():
    cols[name] = fn(O, H, L, C).astype("float64")

ref = pl.DataFrame(cols)
ref.write_parquet(f"{OUT}/talib_reference.parquet")
print(f"talib_reference: rows={ref.height} cols={ref.width} (incl date)")
print("done talib")
