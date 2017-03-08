from math import pi
import numpy as np

from cstool.icdf import icdf


def normal(mu, sigma):
    """Normal distribution."""
    v = sigma**2
    A = (2 * pi * v)**(-0.5)
    return lambda x: A * np.exp(-(mu - x)**2 / (2 * v))


def pdf(x):
    """Example PDF with two peaks."""
    return (normal(-1, 1)(x) + normal(1, 0.5)(x)) / 2.


def test_icdf():
    p = np.linspace(0.0, 1.0, 1025)
    F = icdf(pdf, -5, 5, 1024)
    dp = (p[1:] - p[:-1]) / (F[1:] - F[:-1])
    x = (F[1:] + F[:-1]) / 2.

    err = abs(dp - pdf(x))
    assert max(err) < 1e-3
    assert err.mean() < 1e-4

