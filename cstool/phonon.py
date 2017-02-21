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
    """
    Compute the differential phonon cross-sections given the properties
    of a material. These properties should be given as quantities with units,
    where the unit must have the same dimensionality as those given here.

    :param eps_ac: acoustic deformation potential (eV)
    :param c_s: speed of sound (m/s) unit? 
    :param M: molar weight (g/mol)
    :param rho_m: mass density (g/cm³)
    :param lattice: lattice constant (Å)
    :param E_BZ: the electron energy at the Brioullin zone (eV); can be deduced from `lattice`.
    :return: Function taking an energy array and an angle array, returning the
             crosssection quantity in units of cm² as a 2d-array.

    One of the parameters `lattice` and `E_BZ` should be given.
    """

    # Material related parameters. These will be moved to argument
    alpha_single_branch = 0. * units('m²/s')  # relates to the bending of the dispersion relation towards the Brillouin zone boundary (used in Eq. 3.112)
    m_dos = 1 * units.m_e               # :param m_dos: density of state mass (kg)
    m_effective = 1 * units.m_e         # :param m_effective: effective mass of particle a.k.a. m_star (kg)

    if lattice is None and E_BZ is None:
        raise ValueError("One of `lattice` and `E_BZ` should be given.")

    k_BZ = 2 * pi / lattice # wave factor at 1st Brillouin Zone Boundary
        
    E_BZ = E_BZ or ((units.hbar * k_BZ)**2 / (2 * units.m_e)).to('eV')  # Verduin Eq. 3.120
    lattice = lattice or np.sqrt(units.h**2 / (2 * units.m_e*E_BZ)).to('Å') # If lattice is not given, but E_BZ is defined.

    # print("E_BZ = {:~P}".format(E_BZ.to('eV'))) ??

    A = 5*E_BZ                                  # A: screening factor (eV); 5 is constant for every material.
    rho_n = (units.N_A / M * rho_m).to('cm⁻³')  # rho_n: number density.
    
    h_bar_w_BZ = (units.hbar * c_s *k_BZ - units.hbar * alpha_single_branch * k_BZ**2).to('eV') # Verduin Eq. 3.114
    n_BZ = 1 / (expm1(h_bar_w_BZ / (units.k * T)) - 1) # Acoustic phonon population density , Verduin Eq. 3.117
    
    sigma_ac = ((np.sqrt(m_effective * m_dos**3) * eps_ac**2 * units.k * T) /
                (pi * units.hbar**4 * c_s**2 * rho_m * rho_n)).to('cm²') # Verduin equation (3.125) divided by number density
    # 
    # extra multiplication factor for high energies according to Verduin equation (3.126)
    # noticed that A could be balanced out of the equation
    factor_high = ((n_BZ + 0.5) * 8 * m_dos * c_s**2 / (h_bar_w_BZ * units.k * T)).to('1/eV')
    # alpha = ((n_BZ + 0.5) * 8 * h_bar_w_BZ / (units.k*T * E_BZ)).to('1/eV')

    def mu(theta):  # see Eq. 3.126  
        return (1 - cos(theta)) / 2

    def norm(mu, E):
        """Phonon cross-section for low energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        return (sigma_ac / (4*pi * (1 + mu * E/A)**2)).to('cm²')

    def dcs_hi(mu, E):
        """Phonon cross-section for high energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        return (factor_high * mu * E).to(units.dimensionless)

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
        interpolate=log_interpolate) # , h=lambda x: x)


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
