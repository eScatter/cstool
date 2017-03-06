from cslib.dataframe import DataFrame
from elf import read_elf_data
from cslib import units

from math import pi

import numpy as np
from functools import reduce

def loglog_interpolate(x_i, y_i):
    """Interpolates the tabulated values. Linear interpolation
    on a log-log scale. Requires `y_i` to be unitless."""

    assert y_i.dimensionless, "y_i should be dimensionless"
    assert y_i.shape == x_i.shape, "shapes should match"

    x_log_steps = np.log(x_i[1:]/x_i[:-1])
    log_y_i = np.log(y_i)

    def f(x):
        x_idx = np.searchsorted(x_i.flat, x.to(x_i.units).flat)
        mx_idx = np.ma.array(
            x_idx - 1,
            mask=np.logical_or(x_idx == 0,
                               x_idx == x.size))

        # compute the weight factor
        w = np.log(x / (np.ma.take(x_i, mx_idx) * x_i.units)) \
            / np.ma.take(x_log_steps, mx_idx)

        # take elements from a masked NdArray
        def take(a, *ix):
            i = np.meshgrid(*ix[::-1])[::-1]
            m = reduce(np.logical_or, [j.mask for j in i])
            return np.ma.array(a[[j.filled(0) for j in i]], mask=m)

        y = (1 - w) * take(log_y_i, mx_idx) \
            + w * take(log_y_i, mx_idx + 1)

        return np.exp(y).filled(0) * y_i.units

    return f

def f_sumrule(filename, M, rho_m, n = 1):
    """ELF data has to obey the f-sum rule (Abril 1998, eq. 8)

    This function takes the ELF, the molar mass, the atomic density and the
    amount of atoms per molecule and returns the value of the f-sum rule."""
    elf_data = read_elf_data(filename)
    elf = loglog_interpolate(elf_data['w0'], elf_data['elf'])

    print(elf_data)
    print()
    print()
    w = np.logspace(-1, 3, 1024) * units.eV
    e = elf(w)
    N = (rho_m * units.N_A * n / M).to('cm⁻³') # atomic density
    fsum = 1. / (2 * pi**2 * N) * sum((e[:-1] + e[1:]) / 2 * (w[:-1] + w[1:]) /
        2 * (w[1:] - w[:-1]))
    return fsum

if __name__ == "__main__":
    import sys
    from cstool.parse_input import read_input
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate f-sum rule for a material.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in YAML format.")
    args = parser.parse_args()

    s = read_input(args.material_file)
    if 'M_tot' not in s:
        s.M_tot = sum(e.M * e.count for e in s.elements.values())
    if 'n' not in s:
        s.n = sum(e.count for e in s.elements.values())

    fsum = f_sumrule(s.elf_file, s.M_tot, s.rho_m, s.n)
    print(fsum.to('m**3*J**2'))
