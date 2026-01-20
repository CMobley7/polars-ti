# -*- coding: utf-8 -*-
from numpy import array, inf, log, nan, pi, sqrt

from polars_ti._typing import Array, IntFloat, Number
from polars_ti.maps import Imports
from polars_ti.utils import hpoly


def _gaussian_poly_coefficients() -> Array:
    """Three pairs of Polynomial Approximation Coefficients
    for the Gaussian Normal CDF"""

    p0 = array(
        [
            -5.99633501014107895267e1,
            9.80010754185999661536e1,
            -5.66762857469070293439e1,
            1.39312609387279679503e1,
            -1.23916583867381258016e0,
        ]
    )
    q0 = array(
        [
            1.00000000000000000000e0,
            1.95448858338141759834e0,
            4.67627912898881538453e0,
            8.63602421390890590575e1,
            -2.25462687854119370527e2,
            2.00260212380060660359e2,
            -8.20372256168333339912e1,
            1.59056225126211695515e1,
            -1.18331621121330003142e0,
        ]
    )

    p1 = array(
        [
            4.05544892305962419923e0,
            3.15251094599893866154e1,
            5.71628192246421288162e1,
            4.40805073893200834700e1,
            1.46849561928858024014e1,
            2.18663306850790267539e0,
            -1.40256079171354495875e-1,
            -3.50424626827848203418e-2,
            -8.57456785154685413611e-4,
        ]
    )
    q1 = array(
        [
            1.00000000000000000000e0,
            1.57799883256466749731e1,
            4.53907635128879210584e1,
            4.13172038254672030440e1,
            1.50425385692907503408e1,
            2.50464946208309415979e0,
            -1.42182922854787788574e-1,
            -3.80806407691578277194e-2,
            -9.33259480895457427372e-4,
        ]
    )

    p2 = array(
        [
            3.23774891776946035970e0,
            6.91522889068984211695e0,
            3.93881025292474443415e0,
            1.33303460815807542389e0,
            2.01485389549179081538e-1,
            1.23716634817820021358e-2,
            3.01581553508235416007e-4,
            2.65806974686737550832e-6,
            6.23974539184983293730e-9,
        ]
    )
    q2 = array(
        [
            1.00000000000000000000e0,
            6.02427039364742014255e0,
            3.67983563856160859403e0,
            1.37702099489081330271e0,
            2.16236993594496635890e-1,
            1.34204006088543189037e-2,
            3.28014464682127739104e-4,
            2.89247864745380683936e-6,
            6.79019408009981274425e-9,
        ]
    )

    return [p0, q0, p1, q1, p2, q2]


def inv_norm(value: IntFloat) -> Number | None:
    """Inverse Normal (inv_norm)
    Calculates the 'x' in which the area under the Gaussian PDF is
    equal to value.

    If the user has package "statsmodels" installed, the method will call and
    return norm().ppf(value)

    Source: https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/scipy/special/cephes/ndtri.c
    """

    if Imports["statsmodels"]:
        from scipy.stats import norm

        return norm().ppf(value)

    negate = True
    v = value

    if v == 0.0:
        return -inf
    if v == 1.0:
        return inf
    if v < 0.0 or value > 1.0:
        return nan

    p0, q0, p1, q1, p2, q2 = _gaussian_poly_coefficients()

    sqrt2pi = sqrt(2 * pi)
    threshold = 0.13533528323661269189
    if v > 1.0 - threshold:
        v, negate = 1.0 - v, False

    # 0 <= |x0 - 0.5| <= 3/8
    if v > threshold:
        v -= 0.5
        v2 = v * v
        y = v + v * (v2 * hpoly(p0, v2) / hpoly(q0, v2))
        y *= sqrt2pi
        return y

    y = sqrt(-2.0 * log(v))
    y0 = y - log(y) / y

    z = 1.0 / y
    if y < 8.0:
        # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
        #  i.e., x between exp(-2) = .135 and exp(-32) = 1.27e-14.
        y1 = z * hpoly(p1, z) / hpoly(q1, z)
    else:
        # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
        # i.e., x between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
        y1 = z * hpoly(p2, z) / hpoly(q2, z)

    y = y0 - y1
    if negate:
        y = -y

    return y
