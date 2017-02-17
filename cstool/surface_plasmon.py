# from cslib import units

import numpy as np
from .log_interpolate import log_interpolate_1d


def maclaurin_integral(f, a, b, n, x):
    def trapezium(g, xi, xj):
        """Trapezium rule

        :param wi: should be column vector
        :param wj: is row vector of either odd or even indices."""
        return (g/(xj - xi) + g/(xj + xi)) / 2

    xi, h = np.linspace(a, b, n, endpoint=True, retstep=True)
    fi = f(xi)

    # We want to compute at the points in `x`; to be able to interpolate the
    # value at each `x`, we need the two neighbouring points on the sampled
    # values `xi`.
    # The lower index of that interval is stored in `i0`, the upper index in
    # `i1`. To see at which points we need to compute the integral we find the
    # union of all points in `i0` and `i1`.
    # The indices in `idx` are the places in the `xi` array.
    i = np.searchsorted(xi, x)
    i0 = np.where(i - 1 < 0, 0, i - 1)
    i1 = np.where(i == 0, 1, np.where(i >= n, n-1, i))
    idx = np.unique(np.r_[i0, i1])

    # Next we split up the indices in even and odd ones.
    evens = np.where(idx % 2 == 0)
    odds = np.where(idx % 2 != 0)
    i_even = idx[evens]
    i_odd = idx[odds]

    # i odd, j even
    Fj_even = trapezium(
            fi[0::2, np.newaxis],
            xi[np.newaxis, i_odd],
            xi[0::2, np.newaxis])
    # i even, j odd
    Fj_odd = trapezium(
            fi[1::2, np.newaxis],
            xi[np.newaxis, i_even],
            xi[1::2, np.newaxis])

    # evens and odds make idx, so all F is assigned
    F = np.zeros(shape=idx.shape, dtype='float64')
    F[evens] = 2*h * Fj_odd.sum(axis=0)
    F[odds] = 2*h * Fj_even.sum(axis=0)
    return xi[idx], F


def kramers_kronig(W, ELF, n):
    """Compute Kramer's Kronig using:
        Ohta, Koji, and Hatsuo Ishida - "Comparison among several numerical
        integration methods for Kramers-Kronig transformation." Applied
        Spectroscopy 42.6 (1988): 952-957."""
    log_elf_fn = log_interpolate_1d(W, np.log(ELF))

    def elf_fn(wi):
        return np.exp(log_elf_fn(wi))

    W_i, F = maclaurin_integral(elf_fn, W[0], W[-1], n, W)
    G = elf_fn(W_i)

    F = 2/np.pi * F - 1
    U = -F/(F**2 + G**2)        # real part of the dielectric function
    V = G/(F**2 + G**2)         # imaginary part of the dielectric function
    bulk_i = V/(U**2 + V**2)    # the bulk ELF

    # the surface ELF at the interpolated energies
    surface_elf_i = V/((U + 1)**2 + V**2)

    surface_elf = np.exp(log_interpolate_1d(W_i, np.log(surface_elf_i))(W))
    bulk = np.exp(log_interpolate_1d(W_i, np.log(bulk_i))(W))
    return surface_elf, bulk


if __name__ == "__main__":
    def importdata(filename, skip=1):
        data = np.loadtxt(filename, skiprows=skip)
        energy = data[:-1, 0]
        elf = data[:-1, 1]
        return [energy, elf]

    def print_table(x, y):
        for l in np.c_[x, y]:
            print(" ".join(str(i) for i in l))
        print("\n\n")

    [W, ELF] = importdata('./data/elf/df_Si.dat')
    surface_elf, bulk = kramers_kronig(W, ELF, 1000000)

    print_table(W, ELF)
    print_table(W, bulk)
    print_table(W, surface_elf)
