cimport icdf
import numpy as npp
cimport numpy as np

cdef float callback(float x, void *data):
    return (<object>data)(x)

cpdef inverse_cdf(f, float a, float b, unsigned n, float eps=1e-6):
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
    data = npp.zeros(n+1, dtype='float32')
    cdef float [:] data_view = data
    compute_icdf(callback, a, b, n, eps, &data_view[0], <void *>f)
    return data

