# Indicators

Polars-TI provides **267 indicators and candlestick-pattern groups** across 10 categories. Each is available as a Polars expression (`from polars_ti.<category> import <name>`) and via the `df.ti.<name>()` accessor. The **TA-Lib** column marks indicators that accept a `talib=` flag to select the TA-Lib path (default) or the native Polars/Numba path; see [TA-Lib & native paths](talib.md).

> **Outputs** is the number of result columns. Multi-output indicators return a Polars struct (via the accessor) or a list of expressions; see [Getting started](getting-started.md#multi-output-indicators).


## Candles (65)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `cdl_2crows` | 1 | ✓ | Candle Pattern - Two Crows. |
| `cdl_3blackcrows` | 1 | ✓ | Candle Pattern - Three Black Crows. |
| `cdl_3inside` | 1 | ✓ | Candle Pattern - Three Inside Up/Down. |
| `cdl_3linestrike` | 1 | ✓ | Candle Pattern - Three-Line Strike. |
| `cdl_3outside` | 1 | ✓ | Candle Pattern - Three Outside Up/Down. |
| `cdl_3starsinsouth` | 1 | ✓ | Candle Pattern - Three Stars In The South. |
| `cdl_3whitesoldiers` | 1 | ✓ | Candle Pattern - Three Advancing White Soldiers. |
| `cdl_abandonedbaby` | 1 | ✓ | Candle Pattern - Abandoned Baby. |
| `cdl_advanceblock` | 1 | ✓ | Candle Pattern - Advance Block. |
| `cdl_belthold` | 1 | ✓ | Candle Pattern - Belt-hold. |
| `cdl_breakaway` | 1 | ✓ | Candle Pattern - Breakaway. |
| `cdl_closingmarubozu` | 1 | ✓ | Candle Pattern - Closing Marubozu. |
| `cdl_concealbabyswall` | 1 | ✓ | Candle Pattern - Concealing Baby Swallow. |
| `cdl_counterattack` | 1 | ✓ | Candle Pattern - Counterattack. |
| `cdl_darkcloudcover` | 1 | ✓ | Candle Pattern - Dark Cloud Cover. |
| `cdl_doji` | 1 | ✓ | Candle Pattern - Doji. |
| `cdl_dojistar` | 1 | ✓ | Candle Pattern - Doji Star. |
| `cdl_dragonflydoji` | 1 | ✓ | Candle Pattern - Dragonfly Doji. |
| `cdl_engulfing` | 1 | ✓ | Candle Pattern - Engulfing. |
| `cdl_eveningdojistar` | 1 | ✓ | Candle Pattern - Evening Doji Star. |
| `cdl_eveningstar` | 1 | ✓ | Candle Pattern - Evening Star. |
| `cdl_gapsidesidewhite` | 1 | ✓ | Candle Pattern - Gap Side-by-Side White Lines. |
| `cdl_gravestonedoji` | 1 | ✓ | Candle Pattern - Gravestone Doji. |
| `cdl_hammer` | 1 | ✓ | Candle Pattern - Hammer. |
| `cdl_hangingman` | 1 | ✓ | Candle Pattern - Hanging Man. |
| `cdl_harami` | 1 | ✓ | Candle Pattern - Harami. |
| `cdl_haramicross` | 1 | ✓ | Candle Pattern - Harami Cross. |
| `cdl_highwave` | 1 | ✓ | Candle Pattern - High-Wave Candle. |
| `cdl_hikkake` | 1 | ✓ | Candle Pattern - Hikkake. |
| `cdl_hikkakemod` | 1 | ✓ | Candle Pattern - Modified Hikkake. |
| `cdl_homingpigeon` | 1 | ✓ | Candle Pattern - Homing Pigeon. |
| `cdl_identical3crows` | 1 | ✓ | Candle Pattern - Identical Three Crows. |
| `cdl_inneck` | 1 | ✓ | Candle Pattern - In-Neck. |
| `cdl_inside` | 1 | — | Candle Pattern - Inside Bar (no TA-Lib equivalent). |
| `cdl_invertedhammer` | 1 | ✓ | Candle Pattern - Inverted Hammer. |
| `cdl_kicking` | 1 | ✓ | Candle Pattern - Kicking. |
| `cdl_kickingbylength` | 1 | ✓ | Candle Pattern - Kicking By Length. |
| `cdl_ladderbottom` | 1 | ✓ | Candle Pattern - Ladder Bottom. |
| `cdl_longleggeddoji` | 1 | ✓ | Candle Pattern - Long Legged Doji. |
| `cdl_longline` | 1 | ✓ | Candle Pattern - Long Line Candle. |
| `cdl_marubozu` | 1 | ✓ | Candle Pattern - Marubozu. |
| `cdl_matchinglow` | 1 | ✓ | Candle Pattern - Matching Low. |
| `cdl_mathold` | 1 | ✓ | Candle Pattern - Mat Hold. |
| `cdl_morningdojistar` | 1 | ✓ | Candle Pattern - Morning Doji Star. |
| `cdl_morningstar` | 1 | ✓ | Candle Pattern - Morning Star. |
| `cdl_onneck` | 1 | ✓ | Candle Pattern - On-Neck. |
| `cdl_pattern` | 62 | ✓ | Candlestick Pattern Detection. |
| `cdl_piercing` | 1 | ✓ | Candle Pattern - Piercing. |
| `cdl_rickshawman` | 1 | ✓ | Candle Pattern - Rickshaw Man. |
| `cdl_risefall3methods` | 1 | ✓ | Candle Pattern - Rising/Falling Three Methods. |
| `cdl_separatinglines` | 1 | ✓ | Candle Pattern - Separating Lines. |
| `cdl_shootingstar` | 1 | ✓ | Candle Pattern - Shooting Star. |
| `cdl_shortline` | 1 | ✓ | Candle Pattern - Short Line Candle. |
| `cdl_spinningtop` | 1 | ✓ | Candle Pattern - Spinning Top. |
| `cdl_stalledpattern` | 1 | ✓ | Candle Pattern - Stalled Pattern. |
| `cdl_sticksandwich` | 1 | ✓ | Candle Pattern - Stick Sandwich. |
| `cdl_takuri` | 1 | ✓ | Candle Pattern - Takuri. |
| `cdl_tasukigap` | 1 | ✓ | Candle Pattern - Tasuki Gap. |
| `cdl_thrusting` | 1 | ✓ | Candle Pattern - Thrusting. |
| `cdl_tristar` | 1 | ✓ | Candle Pattern - Tristar. |
| `cdl_unique3river` | 1 | ✓ | Candle Pattern - Unique Three River. |
| `cdl_upsidegap2crows` | 1 | ✓ | Candle Pattern - Upside Gap Two Crows. |
| `cdl_xsidegap3methods` | 1 | ✓ | Candle Pattern - Upside/Downside Gap Three Methods. |
| `cdl_z` | 4 | — | Candle Type Z - Rolling Z-Score normalized OHLC |
| `ha` | 4 | — | Heikin-Ashi Candles |

## Cycles (9)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `dsp` | 1 | — | Detrended Synthetic Price (DSP) |
| `ebsw` | 1 | — | Even Better SineWave (EBSW) |
| `ht_dcperiod` | 1 | ✓ | Hilbert Transform Dominant Cycle Period (HT_DCPERIOD) |
| `ht_dcphase` | 1 | ✓ | Hilbert Transform Dominant Cycle Phase (HT_DCPHASE) |
| `ht_phasor` | 2 | ✓ | Hilbert Transform Phasor Components (HT_PHASOR) |
| `ht_sine` | 2 | ✓ | Hilbert Transform Sine Wave (HT_SINE) |
| `ht_trendmode` | 1 | ✓ | Hilbert Transform Trend vs Cycle Mode (HT_TRENDMODE) |
| `msw` | 2 | — | Mesa Sine Wave (MSW) |
| `reflex` | 1 | — | Reflex Indicator |

## Momentum (55)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `ao` | 1 | — | Awesome Oscillator (AO) |
| `apo` | 1 | ✓ | Absolute Price Oscillator (APO) |
| `bias` | 1 | — | Bias (BIAS) |
| `bop` | 1 | ✓ | Balance of Power (BOP) |
| `brar` | 2 | — | BRAR (BR and AR) |
| `cci` | 1 | ✓ | Commodity Channel Index (CCI) |
| `cfo` | 1 | ✓ | Chande Forecast Oscillator (CFO) |
| `cg` | 1 | — | Center of Gravity (CG) |
| `cmo` | 1 | ✓ | Chande Momentum Oscillator (CMO) |
| `coppock` | 1 | — | Coppock Curve (COPC) |
| `crsi` | 1 | ✓ | Connors Relative Strength Index (CRSI) |
| `cti` | 1 | — | Correlation Trend Indicator (CTI) |
| `dm` | 2 | ✓ | Directional Movement (DM) |
| `er` | 1 | — | Efficiency Ratio (ER) |
| `eri` | 2 | — | Elder Ray Index (ERI) |
| `exhc` | 2 | — | Exhaustion Count (EXHC) |
| `fisher` | 2 | — | Fisher Transform (FISHT) |
| `fosc` | 1 | ✓ | Forecast Oscillator (FOSC) |
| `imi` | 1 | — | Intraday Momentum Index (IMI) |
| `inertia` | 1 | ✓ | Inertia (INERTIA) |
| `kdj` | 3 | — | KDJ |
| `kst` | 2 | — | Know Sure Thing (KST) |
| `lrsi` | 1 | — | Laguerre RSI (LRSI) |
| `macd` | 3 | ✓ | Moving Average Convergence Divergence (MACD) |
| `mom` | 1 | ✓ | Momentum (MOM) |
| `pgo` | 1 | ✓ | Pretty Good Oscillator (PGO) |
| `po` | 1 | ✓ | Projection Oscillator (PO) |
| `ppo` | 3 | ✓ | Percentage Price Oscillator (PPO) |
| `psl` | 1 | — | Psychological Line (PSL) |
| `qqe` | 4 | ✓ | Quantitative Qualitative Estimation (QQE) |
| `rmi` | 1 | — | Relative Momentum Index (RMI) |
| `roc` | 1 | ✓ | Rate of Change (ROC) |
| `rocp` | 1 | ✓ | Rate of Change Percentage (ROCP) |
| `rocr` | 1 | ✓ | Rate of Change Ratio (ROCR) |
| `rocr100` | 1 | ✓ | Rate of Change Ratio * 100 (ROCR100) |
| `rsi` | 1 | ✓ | Relative Strength Index (RSI) |
| `rsx` | 1 | — | Relative Strength Xtra (RSX) |
| `rvgi` | 2 | — | Relative Vigor Index (RVGI) |
| `slope` | 1 | — | Slope |
| `smc` | 7 | ✓ | Smart Money Concept (SMC) |
| `smc_sweep` | 1 | — | Smart Money Concept Liquidity Sweep (SMC_SWEEP) |
| `smi` | 3 | ✓ | SMI Ergodic Indicator (SMI) |
| `squeeze` | 4 | — | Squeeze (SQZ) |
| `squeeze_pro` | 6 | — | Squeeze PRO (SQZPRO) |
| `stc` | 3 | — | Schaff Trend Cycle (STC) |
| `stoch` | 3 | ✓ | Stochastic Oscillator (STOCH) |
| `stochf` | 2 | ✓ | Fast Stochastic Oscillator (STOCHF) |
| `stochrsi` | 2 | ✓ | Stochastic RSI (STOCHRSI) |
| `tmo` | 4 | ✓ | True Momentum Oscillator (TMO) |
| `trix` | 2 | ✓ | TRIX (Triple Exponential Average Rate of Change) |
| `trixh` | 3 | ✓ | TRIX Histogram (TRIXH) |
| `tsi` | 2 | ✓ | True Strength Index (TSI) |
| `uo` | 1 | ✓ | Ultimate Oscillator (UO) |
| `vwmacd` | 3 | — | Volume Weighted MACD (VWMACD) |
| `willr` | 1 | ✓ | Williams %R (WILLR) |

## Overlap (44)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `alligator` | 3 | — | Bill Williams Alligator |
| `alma` | 1 | — | Arnaud Legoux Moving Average (ALMA) |
| `avgprice` | 1 | ✓ | Average Price (AVGPRICE) |
| `dema` | 1 | ✓ | Double Exponential Moving Average (DEMA) |
| `ema` | 1 | ✓ | Exponential Moving Average (EMA) |
| `fwma` | 1 | — | Fibonacci's Weighted Moving Average (FWMA) |
| `hilo` | 3 | ✓ | Gann HiLo Activator |
| `hl2` | 1 | ✓ | HL2 - Midpoint of High and Low |
| `hlc3` | 1 | ✓ | HLC3 - Typical Price (Average of High, Low, Close) |
| `hma` | 1 | — | Hull Moving Average (HMA) |
| `hwma` | 1 | — | HWMA (Holt-Winter Moving Average) |
| `ichimoku` | 5 | — | Ichimoku Kinkō Hyō |
| `jma` | 1 | — | Jurik Moving Average (JMA) |
| `kama` | 1 | ✓ | Kaufman's Adaptive Moving Average (KAMA) |
| `linreg` | 1 | ✓ | Linear Regression Moving Average (LINREG) |
| `mama` | 2 | ✓ | Ehler's MESA Adaptive Moving Average (MAMA) |
| `mavp` | 1 | ✓ | Moving Average with Variable Period (MAVP) |
| `mcgd` | 1 | — | McGinley Dynamic Indicator |
| `medprice` | 1 | ✓ | Median Price (MEDPRICE) |
| `midpoint` | 1 | ✓ | Midpoint |
| `midprice` | 1 | ✓ | Midprice (average of rolling high and low) |
| `mmar` | 6 | — | Madrid Moving Average Ribbon (MMAR) |
| `ohlc4` | 1 | ✓ | OHLC4 - Average of Open, High, Low, Close |
| `ott` | 3 | ✓ | Optimized Trend Tracker (OTT) |
| `pivots` | 9 | — | Pivot Points (Pure Polars + Numba) |
| `pwma` | 1 | — | Pascal's Weighted Moving Average (PWMA) |
| `rainbow` | 10 | — | Rainbow Charts |
| `rma` | 1 | — | Wilder's Moving Average (RMA) |
| `sinwma` | 1 | — | Sine Weighted Moving Average (SINWMA) |
| `sma` | 1 | ✓ | Simple Moving Average (SMA) |
| `smma` | 1 | ✓ | SMoothed Moving Average (SMMA) |
| `ssf` | 1 | — | Ehler's Super Smoother Filter (SSF) |
| `ssf3` | 1 | — | Ehler's 3 Pole Super Smoother Filter (SSF3) |
| `supertrend` | 4 | ✓ | Supertrend - uses HL2 and ATR composition. |
| `swma` | 1 | — | Symmetric Weighted Moving Average (SWMA) |
| `t3` | 1 | ✓ | Tim Tillson's T3 Moving Average (T3) |
| `tema` | 1 | ✓ | Triple Exponential Moving Average (TEMA) |
| `trima` | 1 | ✓ | Triangular Moving Average (TRIMA) |
| `tsf` | 1 | ✓ | Time Series Forecast (TSF) |
| `typprice` | 1 | ✓ | Typical Price (TYPPRICE) |
| `vidya` | 1 | ✓ | Variable Index Dynamic Average (VIDYA) |
| `wcp` | 1 | ✓ | Weighted Closing Price (WCP) |
| `wma` | 1 | ✓ | Weighted Moving Average (WMA) |
| `zlma` | 1 | ✓ | Zero Lag Moving Average (ZLMA) |

## Performance (3)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `drawdown` | 3 | — | Drawdown (DD) |
| `log_return` | 1 | — | Log Return |
| `percent_return` | 1 | — | Percent Return |

## Statistics (14)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `beta` | 1 | ✓ | Beta (BETA) |
| `correl` | 1 | ✓ | Pearson Correlation Coefficient (CORREL) |
| `entropy` | 1 | — | Entropy (ENTP) |
| `kurtosis` | 1 | — | Rolling Kurtosis |
| `mad` | 1 | — | Rolling Mean Absolute Deviation |
| `md` | 1 | — | Mean Deviation (MD) — tulipy name: MD |
| `median` | 1 | — | Rolling Median |
| `quantile` | 1 | — | Rolling Quantile |
| `skew` | 1 | — | Rolling Skew |
| `stderr` | 1 | — | Standard Error (STDERR) |
| `stdev` | 1 | ✓ | Rolling Standard Deviation |
| `tos_stdevall` | 7 | — | TOS Standard Deviation All |
| `variance` | 1 | ✓ | Rolling Variance |
| `zscore` | 1 | — | Rolling Z Score |

## Transform (3)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `cube` | 2 | — | Cube Transform |
| `ifisher` | 2 | — | Inverse Fisher Transform |
| `remap` | 1 | — | ReMap (Linear Normalization) |

## Trend (27)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `adx` | 4 | ✓ | Average Directional Index (ADX) |
| `adxr` | 1 | ✓ | Average Directional Movement Index Rating (ADXR) |
| `alphatrend` | 2 | ✓ | Alpha Trend |
| `amat` | 2 | — | Archer Moving Averages Trends (AMAT) |
| `aroon` | 3 | ✓ | Aroon & Aroon Oscillator |
| `chop` | 1 | ✓ | Choppiness Index (CHOP) |
| `cksp` | 2 | ✓ | Chande Kroll Stop (CKSP) |
| `decay` | 1 | — | Decay |
| `decreasing` | 1 | — | Decreasing |
| `dpo` | 1 | — | Detrend Price Oscillator (DPO) |
| `dx` | 1 | ✓ | Directional Index (DX) |
| `ht_trendline` | 1 | ✓ | Hilbert Transform TrendLine (HT_TL) |
| `increasing` | 1 | — | Increasing |
| `long_run` | 1 | — | Long Run |
| `pmax` | 4 | ✓ | PMAX (Price Max) |
| `psar` | 4 | ✓ | Parabolic Stop and Reverse (PSAR) |
| `qstick` | 1 | — | Q Stick |
| `rwi` | 2 | ✓ | Random Walk Index (RWI) |
| `short_run` | 1 | — | Short Run |
| `trama` | 1 | — | Trend Regulated Adaptive Moving Average (TRAMA) |
| `trendflex` | 1 | — | TrendFlex |
| `tsignals` | 4 | — | Trend Signals |
| `ttm_trend` | 1 | — | TTM Trend |
| `vhf` | 1 | — | Vertical Horizontal Filter (VHF) |
| `vortex` | 2 | — | Vortex Indicator |
| `xsignals` | 4 | — | Cross Signals (XSIGNALS) |
| `zigzag` | 3 | — | ZigZag |

## Volatility (22)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `aberration` | 4 | ✓ | Aberration (ABER) |
| `accbands` | 3 | ✓ | Acceleration Bands (ACCBANDS) |
| `atr` | 1 | ✓ | Average True Range (ATR) |
| `atrts` | 1 | ✓ | ATR Trailing Stop (ATRTS) |
| `avolume` | 1 | — | Annualised Historical Volatility (AVOLUME / tulipy: VOLATILITY) |
| `avsl` | 1 | — | Anti-Volume Stop Loss (AVSL) |
| `bbands` | 5 | ✓ | Bollinger Bands (BBANDS) |
| `chandelier_exit` | 3 | ✓ | Chandelier Exit (CHDLREXT) |
| `cvi` | 1 | — | Chaikins Volatility (CVI) |
| `donchian` | 3 | — | Donchian Channels (DC) |
| `fvg` | 3 | — | Fair Value Gap (FVG) |
| `halftrend` | 6 | ✓ | HalfTrend Indicator |
| `hvol` | 1 | — | Historical Volatility (HVOL) |
| `hwc` | 3 | — | HWC (Holt-Winter Channel) |
| `kc` | 3 | ✓ | Keltner Channels (KC) |
| `massi` | 1 | — | Mass Index (MASSI) |
| `natr` | 1 | ✓ | Normalized Average True Range (NATR) |
| `pdist` | 1 | — | Price Distance (PDIST) |
| `rvi` | 1 | ✓ | Relative Volatility Index (RVI) |
| `thermo` | 4 | ✓ | Elders Thermometer (THERMO) |
| `true_range` | 1 | ✓ | True Range |
| `ui` | 1 | — | Ulcer Index (UI) |

## Volume (25)

| Function | Outputs | TA-Lib | Description |
| :--- | :---: | :---: | :--- |
| `ad` | 1 | ✓ | Accumulation/Distribution (AD) |
| `adosc` | 1 | ✓ | Accumulation/Distribution Oscillator (ADOSC) / Chaikin Oscillator |
| `aobv` | 7 | ✓ | Archer On Balance Volume (AOBV) |
| `avwap` | 2 | — | Anchored Volume Weighted Average Price (AVWAP) |
| `cmf` | 1 | — | Chaikin Money Flow (CMF) |
| `efi` | 1 | ✓ | Elder's Force Index (EFI) |
| `emv` | 1 | — | Ease of Movement (EMV) — raw/Tulip variant |
| `eom` | 1 | — | Ease of Movement (EOM) |
| `kvo` | 2 | ✓ | Klinger Volume Oscillator (KVO) |
| `marketfi` | 1 | — | Market Facilitation Index (MARKETFI) |
| `mfi` | 1 | ✓ | Money Flow Index (MFI) |
| `nvi` | 1 | — | Negative Volume Index (NVI) |
| `obv` | 1 | ✓ | On Balance Volume (OBV) |
| `pvi` | 2 | — | Positive Volume Index (PVI) |
| `pvo` | 3 | ✓ | Percentage Volume Oscillator (PVO) |
| `pvol` | 1 | — | Price-Volume (PVOL) |
| `pvr` | 1 | — | Price Volume Rank (PVR) |
| `pvt` | 1 | — | Price-Volume Trend (PVT) |
| `vfi` | 1 | — | Volume Flow Indicator (VFI) |
| `vhm` | 1 | — | Volume Heatmap (VHM) |
| `vosc` | 1 | — | Volume Oscillator (VOSC) |
| `vwap` | 1 | — | Volume Weighted Average Price (VWAP) |
| `vwma` | 1 | — | Volume Weighted Moving Average (VWMA) |
| `wad` | 1 | — | Williams Accumulation/Distribution (WAD) |
| `wb_tsv` | 3 | — | Time Segmented Value (TSV) |
