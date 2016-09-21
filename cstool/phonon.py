# Based on Schreiber & Fitting
# See /doc/extra/phonon-scattering.lyx

from cslib import units, Settings, DCS
from cslib.numeric import (log_interpolate)

from cstool.parse_input import read_input

from math import pi

import numpy as np

from numpy import (cos, expm1, log10)
from functools import partial


def phonon_crosssection(eps_ac, c_s, M, rho_m,
                        lattice=None, E_BZ=None, T=units.T_room,
                        interpolate=log_interpolate,
                        h=lambda x: (3 - 2*x) * x**2):
    """Compute the differential phonon-crosssections given the properties
    of a material. These properties should be given as quantities with units,
    where the unit must have the same dimensionality as those given here.

    :param eps_ac: accoustic deformation potential (eV)
    :param c_s: speed of sound (km/s)
    :param M: molar weight (g/mol)
    :param rho_m: mass density (g/cm³)
    :param lattice: lattice constant (Å)
    :param E_BZ: Brioullin zone energy (eV), can be deduced from `lattice`.
    :return: Function taking an energy array and an angle array, returning the
        crosssection quantity in units of cm² as a 2d-array.

    One of the parameters `lattice` and `E_BZ` should be given.
    """
    if lattice is None and E_BZ is None:
        raise ValueError("One of `lattice` and `E_BZ` should be given.")

    E_BZ = E_BZ or (units.h**2 / (2*units.m_e * lattice**2)).to('eV')

    # print("E_BZ = {:~P}".format(E_BZ.to('eV')))

    A = 5*E_BZ
    rho_n = (units.N_A / M * rho_m).to('cm⁻³')
    h_bar_w_BZ = units.h * c_s / lattice
    n_BZ = 1 / expm1(h_bar_w_BZ / (units.k * T))
    sigma_ac = ((units.m_e**2 * eps_ac**2 * units.k * T) /
                (units.hbar**4 * c_s**2 * rho_m * rho_n)).to('cm²')

    alpha = ((n_BZ + 0.5) * 4 * h_bar_w_BZ / (units.k*T * E_BZ)).to('1/eV')

    def mu(theta):
        return (1 - cos(theta)) / 2

    def norm(mu, E):
        return (sigma_ac / (4*pi * (1 + mu * E/A)**2)).to('cm²')

    def dcs_hi(mu, E):
        """Phonon cross-section for high energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        return (alpha * mu * E).to(units.dimensionless)

    def dcs(E, theta):
        m = mu(theta)

        g = interpolate(
            lambda E: 1, partial(dcs_hi, m),
            h, E_BZ / 4, E_BZ)

        return g(E) * norm(m, E)

    # should have units of m²/sr
    return dcs


def phonon_cs_fn(s: Settings):
    return phonon_crosssection(
        s.eps_ac, s.c_s, s.M_tot, s.rho_m, s.lattice,
        interpolate=log_interpolate)  # , h=lambda x: x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate elastic phonon cross-sections for a material.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in JSON format.")
    args = parser.parse_args()

    s = read_input(args.material_file)
    if 'M_tot' not in s:
        s.M_tot = sum(e.M * e.count for e in s.elements.values())

    E_range = np.logspace(log10(0.01), log10(1000), num=100) * units.eV
    theta_range = np.linspace(0, pi, num=100) * units.rad

    csf = phonon_cs_fn(s)
    cs = DCS(E_range[:, None], theta_range, csf(E_range[:, None], theta_range))

    cs.save_gnuplot('{}_phonon.bin'.format(s.name))
