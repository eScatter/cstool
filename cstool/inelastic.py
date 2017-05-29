from .elf import read_elf_data

from cslib import (units, Settings, DCS)
from numpy import (log, sqrt, log10, pi)

import numpy as np
from functools import reduce


def L_Kieft(K, w0, F):
    """Computes electron cross-sections for inelastic scattering from
    optical data. Model from Kieft & Bosch (2008).

    If used with Numpy arrays and two-dimensional output is required,
    use a column vector for `K` and a row vector for `w0`.

    :param K:
        Kinetic energy of electron.
    :param w0:
        ω₀ - transition energy
    :param F:
        Fermi energy
    """
    a = w0 / K          # dimensionless
    #L1 = np.zeros(len(w0))
    #for i, w in enumerate(w0):
    #    a = w/K
    #    # a < 0.5 test to make sure x1 will be a real number:
    #    if a < 0.5:
    #        x1 = 2/a*(1 + sqrt(1 - 2*a)) - 1
    #        # check if K > F and K - F - w0 > 0:
    #        if (K > F) and ((K - F) > w):
    #            x2 = K - F - w     # energy
    #            x3 = K - F + w

    #            L1[i] = 1.5 * log(x1 * x2 / x3)

    # Here we have  added a few test, so that x1, x2 and x3 are always > 0
    # also in the regimes where L1 should be zero. This is necessary,
    # because even when L1 = 0, the definition of L1 still contains the log()
    # if x1, x2 or x3 < 0, this will result in an error. So the tests are used
    # to prevent the calcuation of log(X) with X <= 0
    # returns x1 for a < 0.5 and 1 for x1 > 0.5
    x1 = (2/a*(1 + sqrt(1 - 2*a * (a < 0.5))) - 1) * (a < 0.5) + (a >= 0.5)
    # returns x2 for K - F > w0 and 1 for K - F < w0
    x2 = (K - F - w0) * ((K-F) > w0) + ((K-F) <= w0) * units('eV')
    # returns x3 for K > F and 1 for K < F
    x3 = (K - F + w0) * (K > F) + (K <= F) * units('eV')
    # returns L1 when (a < 0.5), (K-F > w0) and (K > F) and zero otherwise
    L1 = 1.5 * log(x1 * x2 / x3) * (a < 0.5) * ((K-F) > w0) * (K > F)
    L2 = -log(a) * ((K - F) > w0)

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
    'Ashley_wo_ex': L_Ashley_wo_ex
}


def loglog_interpolate(x_i, y_i):
    """Interpolates the tabulated values. Linear interpolation
    on a log-log scale. Requires `y_i` to be unitless.
    Out-of-range behaviour: extrapolation if x is too high, and
    zero if x is too low."""

    assert y_i.dimensionless, "y_i should be dimensionless"
    assert y_i.shape == x_i.shape, "shapes should match"

    x_log_steps = np.log(x_i[1:]/x_i[:-1])
    log_y_i = np.log(y_i)

    def f(x):
        x_idx = np.searchsorted(x_i.flat, x.to(x_i.units).flat)
        mx_idx = np.clip(x_idx - 1, 0, x_i.size - 2)

        # compute the weight factor
        w = np.log(x / np.take(x_i, mx_idx)) \
            / np.take(x_log_steps, mx_idx)

        y = (1 - w) * np.take(log_y_i, mx_idx) \
            + w * np.take(log_y_i, mx_idx + 1)

        # y is extrapolated on both sides. We want extrapolation
        # for high energy (where a power law is expected), but for low
        # energies we don't know anything, so we want no energy loss.
        return (x_idx != 0) * np.exp(y) * y_i.units

    return f


def inelastic_cs_fn(s: Settings, print_bool=False, L_method: str='Kieft'):
    """Returns a function giving differential cross-sections for
    inelastic scattering, based on the data in the ELF files and
    an extrapolation function `L`, for which there are three options:
    `Kieft`, `Ashley_w_ex` and `Ashley_wo_ex`. The ELF data is
    interpolated on a log-log scale using linear interpolation."""
    assert L_method in methods, \
        "L_method should be in {}".format(list(methods.keys()))

    L = methods[L_method]

    elf_data = read_elf_data(s.elf_file, print_bool)
    elf = loglog_interpolate(elf_data['w0'], elf_data['elf'])
    mc2 = units.m_e * units.c**2

    def cs(K, w):
        #err = np.geterr()
        #np.seterr(all='ignore')
        result = elf(w) * L(K, w, s.fermi) \
            / (pi * units.a_0 * s.rho_n) \
            / (1 - 1 / (K/mc2 + 1)**2) / mc2
        #np.seterr(**err)
        return result

    return cs


def inelastic_cs(s: Settings, L_method: str='Kieft', K_bounds=None):
    """Returns a `DCS` frame on a 1024² grid assuming some
    sensible bounds."""
    assert L_method in methods, \
        "L_method should be in {}".format(list(methods.keys()))

    print("Inelastic cross-sections")
    print("========================")

    K_bounds = K_bounds or (s.fermi + 0.1 * units.eV, 1e4 * units.eV)
    print("Bounds: {k[0].magnitude:.2e} - {k[1].magnitude:.2e}"
          " {k[0].units:~P}".format(k=K_bounds))

    K = np.logspace(
        log10(K_bounds[0].to('eV').magnitude),
        log10(K_bounds[1].to('eV').magnitude), 1024) * units.eV

    # if L_method == 'Kieft':
    #    w0_max = K - s.fermi
    # else:
    #    w0_max = K/2

    elf_data = read_elf_data(s.elf_file)

    w = np.logspace(
        log10(elf_data['w0'][0].to('eV').magnitude),
        log10(K_bounds[1].to('eV').magnitude / 2), 1024) * units.eV

    dcs = inelastic_cs_fn(s, L_method)(K[:, None], w)

    return DCS(K, w, dcs)


if __name__ == "__main__":
    import sys
    from . import read_input

    s = read_input(sys.argv[1])

    elf_data = read_elf_data(s.elf_file)
    elf = loglog_interpolate(elf_data['w0'], elf_data['elf'])

    print(elf_data)
    print()
    print()
    w = np.logspace(-1, 3, 1024) * units.eV
    e = elf(w)
    for l in np.c_[w, e]:
        print(' '.join(map(str, l)))

    inelastic_cs(s).save_gnuplot('{}_inelastic.bin'.format(s.name))
