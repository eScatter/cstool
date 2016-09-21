from cslib import units
from scipy.integrate import quad

import numpy as np


def phonon_loss(c_s, a, T):
    """Compute the phonon loss.

    :param c_s:
        Speed of sound.
    :type c_s: [velocity]

    :param a:
        Lattice spacing
    :type a: [length]

    :param T:
        Temperature
    :type T: [temperature]
    """
    # compute everything in SI units
    c_s = c_s.to('m/s').magnitude
    a = a.to('m').magnitude
    kT = (1 * units.k * T).to('J').magnitude
    hbar = (1 * units.hbar).to('J s').magnitude

    def w(k):
        return 2*c_s / a*np.sin(k * a/2)

    def N(k):
        return 1. / np.expm1(hbar * w(k) / kT)

    def x1(k):
        return w(k)*k*k

    def x2(k):
        return (2*N(k) + 1) * k*k

    y1, err1 = quad(x1, 0, np.pi/a)
    y2, err2 = quad(x2, 0, np.pi/a)

    return hbar * y1/y2 * units.J
