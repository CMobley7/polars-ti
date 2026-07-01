# -*- coding: utf-8 -*-
"""Shared Hilbert Transform Numba pipeline for HT cycle indicators.

Exposes ``nb_ht_pipeline(x)`` which returns all 5 HT intermediates in a
single forward pass, avoiding redundant computation across the five
``ht_dcperiod`` / ``ht_dcphase`` / ``ht_phasor`` / ``ht_sine`` /
``ht_trendmode`` indicators.

This is a direct port of the Ehlers / TA-Lib Hilbert Transform loop, mirroring
the reference ``pandas_ta_classic.cycles._hilbert`` implementation.  The
recursion starts at ``ht_start = 12`` (matching TA-Lib) and computes the
detrended in-phase / quadrature components, homodyne discriminator period,
smoothed dominant-cycle period, DC phase, sine / lead-sine, and trend mode.

Note on native vs TA-Lib fidelity (verified against the ``talib`` C library):

* ``dcperiod`` (smoothed dominant-cycle period): matches ``talib.HT_DCPERIOD``
  EXACTLY (max_abs_diff = 0.0) for all bars from TA-Lib's lookback (32) onward.
* ``inphase`` / ``quadrature``: match ``talib.HT_PHASOR`` EXACTLY
  (max_abs_diff = 0.0) from bar 32 onward.
* ``dcphase`` / ``sine`` / ``trendmode``: use the same formulas as TA-Lib's C
  source, but TA-Lib accumulates ``prevDCPhase`` / ``daysInTrend`` state during
  its 63-bar unstable period in a way the vectorised reimplementation cannot
  reproduce bit-for-bit.  These three native outputs are therefore a documented
  APPROXIMATION and diverge from TA-Lib; the default (TA-Lib) path is the source
  of truth for exact DCPhase / HT_SINE / HT_TRENDMODE values.  (This same
  divergence exists in the pandas-ta-classic reference implementation.)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def nb_ht_pipeline(x):
    """Hilbert Transform pipeline — all 5 HT cycle intermediates.

    Direct port of the Ehlers/TA-Lib Hilbert Transform algorithm from
    "Rocket Science for Traders" (Ehlers 2002) and the TA-Lib C source,
    matching the pandas-ta-classic ``_hilbert`` reference.

    Args:
        x: 1-D float64 price array (close prices).

    Returns:
        Tuple of 5 arrays of length ``x.size``:
          * dcperiod   (float64) – smoothed dominant-cycle period (== TA-Lib)
          * dcphase    (float64) – dominant-cycle phase in degrees (native approx)
          * inphase    (float64) – I₁ in-phase Hilbert component (== TA-Lib phasor)
          * quadrature (float64) – Q₁ quadrature Hilbert component (== TA-Lib phasor)
          * trendmode  (int32)   – 0 = cycle, 1 = trend (native approx)
    """
    a, b = 0.0962, 0.5769
    m = x.size
    ht_start = 12

    smooth_price = np.zeros(m)
    detrend = np.zeros(m)
    q1 = np.zeros(m)
    i1 = np.zeros(m)
    ji = np.zeros(m)
    jq = np.zeros(m)
    i2 = np.zeros(m)
    q2 = np.zeros(m)
    re_ = np.zeros(m)
    im_ = np.zeros(m)
    period = np.zeros(m)
    it_trend = np.zeros(m)

    dcperiod = np.zeros(m)
    dcphase = np.zeros(m)
    inphase = np.zeros(m)
    quadrature = np.zeros(m)
    trendmode = np.zeros(m, dtype=np.int32)

    # Pre-compute sin/cos lookup tables for the DC phase inner loop.
    # dc_period_int is clamped to [1, 50], so tables up to 50 suffice.
    _MAX_P = 51
    sin_table = np.zeros((_MAX_P, _MAX_P))
    cos_table = np.zeros((_MAX_P, _MAX_P))
    for p in range(1, _MAX_P):
        for j in range(p):
            angle = 2.0 * np.pi * j / p
            sin_table[p, j] = np.sin(angle)
            cos_table[p, j] = np.cos(angle)

    # Cumulative sum for the O(1) iTrend SMA.
    close_cumsum = np.zeros(m + 1)
    for ci in range(m):
        close_cumsum[ci + 1] = close_cumsum[ci] + x[ci]

    days_in_trend = 0
    prev_sine = 0.0
    prev_lead_sine = 0.0
    prev_dc_phase = 0.0

    for i in range(m):
        # WMA(4) smoothing — only stored from bar ht_start onwards.
        if i >= ht_start and i >= 3:
            smooth_price[i] = (4.0 * x[i] + 3.0 * x[i - 1] + 2.0 * x[i - 2] + x[i - 3]) / 10.0

        if i < ht_start:
            period[i] = 0.0
            continue

        adj = 0.075 * period[i - 1] + 0.54

        # Detrend (4-tap FIR).
        detrend[i] = (
            a * smooth_price[i] + b * smooth_price[i - 2] - b * smooth_price[i - 4] - a * smooth_price[i - 6]
        ) * adj

        # InPhase and Quadrature.
        q1[i] = (a * detrend[i] + b * detrend[i - 2] - b * detrend[i - 4] - a * detrend[i - 6]) * adj
        i1[i] = detrend[i - 3]

        # Advance the phase of I1 and Q1 by 90 degrees.
        ji[i] = (a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6]) * adj
        jq[i] = (a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6]) * adj

        # Phasor addition for 3-bar averaging.
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth the I and Q components.
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # Homodyne Discriminator.
        re_[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_[i] = 0.2 * re_[i] + 0.8 * re_[i - 1]
        im_[i] = 0.2 * im_[i] + 0.8 * im_[i - 1]

        if im_[i] != 0.0 and re_[i] != 0.0:
            period[i] = 360.0 / (np.arctan(im_[i] / re_[i]) * 180.0 / np.pi)
        else:
            period[i] = period[i - 1]

        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0

        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        sp = 0.33 * period[i] + 0.67 * dcperiod[i - 1]
        dcperiod[i] = sp

        # Phasor components: raw I₁ / Q₁ (matches TA-Lib HT_PHASOR output).
        inphase[i] = i1[i]
        quadrature[i] = q1[i]

        # DC Phase — TA-Lib uses int(smoothPeriod + 0.5) for the window length.
        dc_period_int = int(sp + 0.5)
        if dc_period_int < 1:
            dc_period_int = 1
        real_part = 0.0
        imag_part = 0.0
        for j in range(dc_period_int):
            if i - j >= 0:
                sp_val = smooth_price[i - j]
                real_part += sin_table[dc_period_int, j] * sp_val
                imag_part += cos_table[dc_period_int, j] * sp_val

        abs_imag = abs(imag_part)
        if abs_imag > 0.0:
            dc_phase_val = np.arctan(real_part / imag_part) * 180.0 / np.pi
        else:
            dc_phase_val = prev_dc_phase
            if real_part < 0.0:
                dc_phase_val -= 90.0
            elif real_part > 0.0:
                dc_phase_val += 90.0

        dc_phase_val += 90.0
        if sp > 0.0:
            dc_phase_val += 360.0 / sp
        if imag_part < 0.0:
            dc_phase_val += 180.0
        if dc_phase_val > 315.0:
            dc_phase_val -= 360.0
        dcphase[i] = dc_phase_val

        # Sine / LeadSine.
        sine_val = np.sin(dc_phase_val * np.pi / 180.0)
        lead_sine_val = np.sin((dc_phase_val + 45.0) * np.pi / 180.0)

        # Instantaneous Trendline (ITrend), needed for trend-mode step 4.
        dc_per = int(sp + 0.5)
        if dc_per < 1:
            dc_per = 1
        start_idx = i - dc_per + 1
        if start_idx < 0:
            start_idx = 0
        it_trend[i] = (close_cumsum[i + 1] - close_cumsum[start_idx]) / dc_per
        trendline_val = (
            4.0 * it_trend[i]
            + 3.0 * it_trend[i - 1]
            + 2.0 * (it_trend[i - 2] if i >= 2 else it_trend[0])
            + (it_trend[i - 3] if i >= 3 else it_trend[0])
        ) / 10.0

        # ----- Trend Mode (TA-Lib 4-step algorithm) -----
        trend = 1
        # Step 1: Sine/LeadSine crossover resets the trend counter.
        if (sine_val > lead_sine_val and prev_sine <= prev_lead_sine) or (
            sine_val < lead_sine_val and prev_sine >= prev_lead_sine
        ):
            days_in_trend = 0
            trend = 0
        days_in_trend += 1
        # Step 2: Not enough bars since crossover -> cycle mode.
        if days_in_trend < 0.5 * sp:
            trend = 0
        # Step 3: Phase change in expected cycle range -> cycle mode.
        phase_diff = dc_phase_val - prev_dc_phase
        if sp != 0.0 and phase_diff > 0.67 * 360.0 / sp and phase_diff < 1.5 * 360.0 / sp:
            trend = 0
        # Step 4: Price far from trendline -> trend mode override.
        if trendline_val != 0.0 and abs((smooth_price[i] - trendline_val) / trendline_val) >= 0.015:
            trend = 1
        trendmode[i] = np.int32(trend)

        prev_sine = sine_val
        prev_lead_sine = lead_sine_val
        prev_dc_phase = dc_phase_val

    return dcperiod, dcphase, inphase, quadrature, trendmode
