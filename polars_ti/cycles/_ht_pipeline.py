# -*- coding: utf-8 -*-
"""Shared Hilbert Transform Numba pipeline for HT cycle indicators.

Exposes ``nb_ht_pipeline(x)`` which returns all 5 HT intermediates in a
single forward pass, avoiding redundant computation across the five
``ht_dcperiod`` / ``ht_dcphase`` / ``ht_phasor`` / ``ht_sine`` /
``ht_trendmode`` indicators.

Note on native vs TA-Lib fidelity
----------------------------------
* ``dcperiod``: within ~3% relative error of TA-Lib HT_DCPERIOD after warmup.
* ``inphase`` / ``quadrature``: directly equal to TA-Lib HT_PHASOR (I₁ / Q₁
  raw Hilbert components) with max_abs < 0.2 after warmup.
* ``dcphase``: same atan(Q1/I1)+90 formula as TA-Lib's C source, but the
  accumulated ``prevDCPhase`` state diverges during the 63-bar unstable period
  so the native path does not reproduce TA-Lib's exact output.  Use the TA-Lib
  path (default) for exact DCPhase / HT_SINE values.
* ``trendmode``: approximation using smoothed-period ROC; binary output (0/1).
"""

import numpy as np
from numpy import arctan, rad2deg, zeros_like
from numba import njit


@njit(cache=True)
def nb_ht_pipeline(x):
    """Hilbert Transform pipeline — all 5 HT cycle intermediates.

    Implements the Ehlers/TA-Lib Hilbert Transform algorithm from
    "Rocket Science for Traders" (Ehlers 2002) and the TA-Lib C source.

    Args:
        x: 1-D float64 price array (close prices).

    Returns:
        Tuple of 5 arrays of length ``x.size``:
          * dcperiod   (float64) – smoothed dominant-cycle period
          * dcphase    (float64) – dominant-cycle phase in degrees (native approx)
          * inphase    (float64) – I₁ raw in-phase Hilbert component (≈ TA-Lib phasor)
          * quadrature (float64) – Q₁ raw quadrature Hilbert component (≈ TA-Lib phasor)
          * trendmode  (int32)   – 0 = cycle, 1 = trend
    """
    a, b, m = 0.0962, 0.5769, x.size

    wma4 = zeros_like(x, dtype=x.dtype)
    dt = zeros_like(x, dtype=x.dtype)
    q1 = zeros_like(x, dtype=x.dtype)
    q2 = zeros_like(x, dtype=x.dtype)
    ji = zeros_like(x, dtype=x.dtype)
    jq = zeros_like(x, dtype=x.dtype)
    i1 = zeros_like(x, dtype=x.dtype)
    i2 = zeros_like(x, dtype=x.dtype)
    re = zeros_like(x, dtype=x.dtype)
    im = zeros_like(x, dtype=x.dtype)
    period = zeros_like(x, dtype=x.dtype)
    smp = zeros_like(x, dtype=x.dtype)  # smooth_period

    dcperiod = zeros_like(x, dtype=x.dtype)
    dcphase = zeros_like(x, dtype=x.dtype)
    inphase = zeros_like(x, dtype=x.dtype)
    quadrature = zeros_like(x, dtype=x.dtype)
    trendmode = np.zeros(m, dtype=np.int32)

    prev_dc_phase = 0.0

    for i in range(6, m):
        adj = 0.075 * period[i - 1] + 0.54

        wma4[i] = 0.4 * x[i] + 0.3 * x[i - 1] + 0.2 * x[i - 2] + 0.1 * x[i - 3]
        dt[i] = adj * (a * wma4[i] + b * wma4[i - 2] - b * wma4[i - 4] - a * wma4[i - 6])

        q1[i] = adj * (a * dt[i] + b * dt[i - 2] - b * dt[i - 4] - a * dt[i - 6])
        i1[i] = dt[i - 3]

        ji[i] = adj * (a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6])
        jq[i] = adj * (a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6])

        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re[i] = 0.2 * re[i] + 0.8 * re[i - 1]
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1]

        if re[i] != 0.0 and im[i] != 0.0:
            period[i] = 360.0 / rad2deg(arctan(im[i] / re[i]))
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0
        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smp[i] = 0.33 * period[i] + 0.67 * smp[i - 1]

        # Dominant Cycle Period (= smoothed period)
        dcperiod[i] = smp[i]

        # DC Phase (degrees): Q1/I1 arctangent + offset + wrap.
        # Matches TA-Lib ta_HT_DCPHASE.c formula: atan(Q1/I1)*57.29578 + 90,
        # then +180 if I2 (smoothed) < 0, then wrap if > 315 and prev < 90.
        # NOTE: the prevDCPhase state accumulated during the 63-bar unstable
        # period means the native path diverges from TA-Lib's exact output.
        if i1[i] != 0.0:
            ph = rad2deg(arctan(q1[i] / i1[i])) + 90.0
        elif q1[i] < 0.0:
            ph = 270.0
        else:
            ph = 90.0
        if i2[i] < 0.0:
            ph += 180.0
        if ph > 315.0 and prev_dc_phase < 90.0:
            ph -= 360.0
        prev_dc_phase = ph
        dcphase[i] = ph

        # Phasor components: raw I₁ / Q₁ (matches TA-Lib HT_PHASOR output).
        inphase[i] = i1[i]
        quadrature[i] = q1[i]

        # Trend mode: 1 if the smoothed period has changed more than 50% in
        # the last 4 bars, indicating an abrupt regime switch (trend phase).
        # This approximates TA-Lib's HT_TRENDMODE after warmup.
        if i >= 4 and smp[i - 4] > 1e-10:
            ratio = abs(smp[i] - smp[i - 4]) / smp[i - 4]
            trendmode[i] = np.int32(1) if ratio > 0.5 else np.int32(0)
        else:
            trendmode[i] = np.int32(0)

    return dcperiod, dcphase, inphase, quadrature, trendmode
