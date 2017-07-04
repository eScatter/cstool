from cslib import (units, Settings, DCS)
from numpy import (log, sqrt, log10, pi)

import numpy as np
from warnings import warn

def L_Kieft(K, w0, F):
    """Computes electron cross-sections for inelastic scattering from
    optical data. Model from Kieft & Bosch (2008).

    Typically called with a scalar parameter for K and a numpy array
    for w0. Returns a numpy array, where each value corresponds to a w0.

    :param K:
        Kinetic energy of electron.
    :param w0:
        ω₀ - transition energy
    :param F:
        Fermi energy
    """

    # For sqrt & log calls, we have to strip the units. pint does not like "where".

    a = w0 / K
    s = sqrt((1 - 2*a).magnitude, where = (a <= .5)) * (a <= .5)

    L1_range = (a > 0) * (a < .5) * (K-F > w0) * (K > F)
    L2_range = (a > 0) * (K-F > w0) * (K > F)

    # Calculate L1
    x1 = 2/a * (1 + s) - 1
    x2 = K - F - w0
    x3 = K - F + w0
    L1 = 1.5 * log((x1 * x2 / x3).magnitude, where = L1_range) * L1_range

    # Calculate L2
    L2 = -log(a.magnitude, where = L2_range) * L2_range

    return np.maximum(0, (w0 < 50 * units.eV) * L1
                      + (w0 > 50 * units.eV) * L2)

def L_dv1(K, w0, F):
    """Computes electron cross-sections for inelastic scattering from
    optical data. Model is conceptually somewhere between L_Kieft and L_Ashley:
    L1 is a Fermi-corrected version of Ashley without the factor 3/2
    rescale by Kieft; L2 is the same as in Kieft.

    :param K:
        Kinetic energy of electron.
    :param w0:
        ω₀ - transition energy
    :param F:
        Fermi energy
    """

    a = (w0 / K).magnitude
    b = (F / K).magnitude
    s = sqrt(1 - 2*a, where = (a <= .5), out = np.zeros(a.shape))

    L1_range = (a > 0) * (a < .5) * (a - s < 1 - 2*b)
    L2_range = (a > 0) * (a < 1 - b)

    # Calculate L1
    wm = (1 + a - s)/2
    wp = np.minimum((1 + a + s)/2, 1 - b)
    L1 = log((wp - a) * wm / (wp * (wm - a)), where = L1_range, out = np.zeros(a.shape))

    # Calculate L2
    L2 = -log(a, where = L2_range, out = np.zeros(a.shape))

    return np.maximum(0, (w0 < 50 * units.eV) * L1
                      + (w0 > 50 * units.eV) * L2)

def L_Ashley_w_ex(K, w0, _):
    a = w0 / K
    return (1 - a) * log(4/a) - 7/4*a + a**(3/2) - 33/32*a**2


def L_Ashley_wo_ex(K, w0, _):
    a = w0 / K
    s = sqrt(1 - 2*a)
    return log((1 - a/2 + s)/(1 - a/2 - s))


methods = {
    'Kieft': L_Kieft,
    'Ashley_w_ex': L_Ashley_w_ex,
    'Ashley_wo_ex': L_Ashley_wo_ex,
    'dv1': L_dv1
}


def inelastic_cs_fn(s: Settings, L_method: str='Kieft'):
    """Returns a function giving differential cross-sections for
    inelastic scattering, based on the data in the ELF files and
    an extrapolation function `L`, for which there are three options:
    `Kieft`, `Ashley_w_ex` and `Ashley_wo_ex`."""
    assert L_method in methods, \
        "L_method should be in {}".format(list(methods.keys()))

    L = methods[L_method]

    mc2 = units.m_e * units.c**2

    def cs(K, w):
        result = s.elf_file(w) * L(K, w, s.band_structure.fermi) \
            / (pi * units.a_0 * s.rho_n) \
            / (1 - 1 / (K/mc2 + 1)**2) / mc2
        return result

    return cs


if __name__ == "__main__":
    import sys
    from .parse_input import read_input
    s = read_input(sys.argv[1])

    w = np.logspace(-1, 3, 1024) * units.eV
    e = s.elf_file(w)
    for l in np.c_[w, e]:
        print(' '.join(map(str, l)))

    inelastic_cs(s).save_gnuplot('{}_inelastic.bin'.format(s.name))
