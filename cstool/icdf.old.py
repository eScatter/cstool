from ctypes import (c_float, c_uint, POINTER, CFUNCTYPE, cdll, util)
import numpy as np


PDFFUNC = CFUNCTYPE(c_float, c_float)
libicdf_path = util.find_library("icdf")

if not libicdf_path:
    raise RuntimeError("Could not find shared library 'icdf'.")

libicdf = cdll.LoadLibrary(libicdf_path)

compute_icdf = libicdf.compute_icdf
compute_icdf.argtypes = [PDFFUNC, c_float, c_float, c_uint, c_float,
                         POINTER(c_float)]
compute_icdf.restype = None


def icdf(pdf, a, b, n, epsilon=1e-6):
    """Tabulates the inverse cumulative distribution function of a given
    probability density function.

    To draw random numbers from a distribution, we need an expression for
    the inverse of the CDF of that distribution.  Often we do not have an
    analytic expression for the CDF. This factory function creates a
    Tabulated object, giving a function with a unit domain and a range from
    `a` to `b`.

    :param pdf:
        the probability density function
    :param a:
        left bound
    :param b:
        right bound
    :param n:
        number of steps in the table, should be power of two.
    :param epsilon:
        absolute precision

    This method uses Romberg integration to compute the CDF and then
    Brent-Newton method to find the inverse. We optimise for fast look-up
    and linear interpolation on a uniform grid. To prevent round-off error
    we first compute the integral over the entire range, and then subdivide.

    The final number of elements in the table will be n + 1, including both
    bounding values."""
    data = np.zeros(n+1, dtype='float32')
    compute_icdf(
        PDFFUNC(pdf), c_float(a), c_float(b), c_uint(n), c_float(epsilon),
        data.ctypes.data_as(POINTER(c_float)))
    return data
