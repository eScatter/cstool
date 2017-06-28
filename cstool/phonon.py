# Based on Schreiber & Fitting
# See /doc/extra/phonon-scattering.lyx

from cslib import units, Settings
from cslib.dcs import DCS
from cslib.numeric import (log_interpolate)

from cstool.parse_input import read_input

from math import pi

import numpy as np

from numpy import (cos, expm1, log10)
from functools import partial


def phonon_crosssection(M, rho_m, eps_ac, c_s, alpha,
                        m_dos, m_eff,
                        lattice=None,
                        E_BZ=None, T=units.T_room,
                        interpolate=log_interpolate,
                        h=lambda x: (3 - 2 * x) * x**2):
    """
    Compute the differential phonon cross-sections using the single branch
    model given the properties of a material. These properties should be
    given as quantities with units, where the unit must have the same
    dimensionality as those given here.

    :param M: molar weight (g/mol)
    :param rho_m: mass density (g/cm³)
    :param eps_ac: acoustic deformation potential (eV)
    :param c_s: speed of sound (m/s) unit?
    :paran alpha: relates to the bending of the dispersion relation towards
    the Brillouin zone boundary (used in Eq. 3.112)('m²/s')
    :param m_dos: density of state mass (kg)
    :param m_eff: effective mass of particle a.k.a. m_star (kg)
    :param lattice: lattice constant (Å)
    :param E_BZ: the electron energy at the Brioullin zone (eV); can be deduced
    from `lattice`.
    :return: Function taking an energy array and an angle array, returning the
             crosssection quantity in units of cm² as a 2d-array.

    One of the parameters `lattice` and `E_BZ` should be given.
    """

    if lattice is None and E_BZ is None:
        raise ValueError("One of `lattice` and `E_BZ` should be given.")

    # wave factor at 1st Brillouin Zone Boundary
    k_BZ = 2 * pi / lattice

    # Verduin Eq. 3.120
    E_BZ = E_BZ or ((units.hbar * k_BZ)**2 / (2 * units.m_e)).to('eV')

    # If lattice is not given, but E_BZ is defined.
    lattice = lattice or np.sqrt(units.h**2 / (2 * units.m_e * E_BZ)).to('Å')

    # print("E_BZ = {:~P}".format(E_BZ.to('eV'))) ??

    # A: screening factor (eV); 5 is constant for every material.
    A = 5 * E_BZ
    # rho_n: number density.
    rho_n = (units.N_A / M * rho_m).to('cm⁻³')

    h_bar_w_BZ = (units.hbar * (c_s * k_BZ - alpha * k_BZ**2)) \
        .to('eV')    # Verduin Eq. 3.114

    # Acoustic phonon population density , Verduin Eq. 3.117
    n_BZ = 1 / expm1(h_bar_w_BZ / (units.k * T))

    # Verduin equation (3.125) divided by number density
    sigma_ac = ((np.sqrt(m_eff * m_dos**3) * eps_ac**2 * units.k * T) /
                (pi * units.hbar**4 * c_s**2 * rho_m * rho_n)).to('cm²')

    # extra multiplication factor for high energies according to Verduin
    # equation (3.126) noticed that A could be balanced out of the equation
    factor_high = ((n_BZ + 0.5) * 8 * m_dos * c_s**2 /
                   (h_bar_w_BZ * units.k * T)).to('1/eV')
    # alpha = ((n_BZ + 0.5) * 8 * h_bar_w_BZ / (units.k*T * E_BZ)).to('1/eV')

    def mu(theta):  # see Eq. 3.126
        return (1 - cos(theta)) / 2

    def norm(mu, E):
        """Phonon cross-section for low energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        return (sigma_ac / (4 * pi * (1 + mu * E / A)**2)).to('cm²')

    def dcs_hi(mu, E):
        """Phonon cross-section for high energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        return (factor_high * mu * E).to(units.dimensionless)

    def dcs(theta, E):
        m = mu(theta)

        g = interpolate(
            lambda E: 1, partial(dcs_hi, m),
            h, E_BZ / 4, E_BZ)

        return g(E) * norm(m, E)

    # should have units of m²/sr
    return dcs


def phonon_crosssection_dual_branch(
        M, rho_m,
        eps_ac_lo, c_s_lo, alpha_lo,
        eps_ac_tr, c_s_tr, alpha_tr,
        m_dos, m_eff,
        lattice=None, E_BZ=None, T=units.T_room,
        interpolate=log_interpolate, h=lambda x: (3 - 2 * x) * x**2):
    """Compute the differential phonon cross-sections using dual-branch model
    given the properties of a material. These properties should be given as
    quantities with units, where the unit must have the same dimensionality as
    those given here.

    :param eps_ac_lo: acoustic deformation potential for longitudinal mode (eV)
    :param c_s_lo: speed of sound for longitudinal mode (m/s)
    :param alpha_lo: relates to the bending of the dispersion relation
    towards the Brillouin zone boundary for longitudinal mode (m²/s)
    :param eps_ac_tr: acoustic deformation potential for transversal mode (eV)
    :param c_s_tr: speed of sound for transversal mode (m/s)
    :param alpha_tr: relates to the bending of the dispersion relation
    towards the Brillouin zone boundary for transversal mode (m²/s)
    :param m_dos: density of states electron mass (kg)
    :param m_eff: effective electron mass inside the solid state (kg)
    :param M: molar weight (g/mol)
    :param rho_m: mass density (g/cm³)
    :param lattice: lattice constant (Å)
    :param E_BZ: the electron energy at the Brioullin zone (eV); can be deduced
    from `lattice`.
    :return: Function taking an energy array and an angle array, returning the
             crosssection quantity in units of cm² as a 2d-array.

    One of the parameters `lattice` and `E_BZ` should be given.
    """

    if lattice is None and E_BZ is None:
        raise ValueError("One of `lattice` and `E_BZ` should be given.")

    # wave factor at 1st Brillouin Zone Boundary
    k_BZ = 2 * pi / lattice

    # Verduin Eq. 3.120
    E_BZ = E_BZ or ((units.hbar * k_BZ)**2 / (2 * units.m_e)).to('eV')

    # If lattice is not given, but E_BZ is defined.
    lattice = lattice or np.sqrt(units.h**2 / (2 * units.m_e * E_BZ)).to('Å')

    # print("E_BZ = {:~P}".format(E_BZ.to('eV'))) ??

    # A: screening factor (eV); 5 is constant for every material.
    A = 5 * E_BZ
    # rho_n: number density.
    rho_n = (units.N_A / M * rho_m).to('cm⁻³')

    h_bar_w_BZ_lo = (units.hbar * (c_s_lo * k_BZ - alpha_lo * k_BZ**2)) \
        .to('eV')    # Verduin Eq. 3.114
    h_bar_w_BZ_tr = (units.hbar * (c_s_tr * k_BZ - alpha_tr * k_BZ**2)) \
        .to('eV')    # Verduin Eq. 3.114

    # Acoustic phonon population density , Verduin Eq. 3.117
    n_BZ_lo = 1 / expm1(h_bar_w_BZ_lo / (units.k * T))
    n_BZ_tr = 1 / expm1(h_bar_w_BZ_tr / (units.k * T))

    # Verduin equation (3.125) divided by number density without branch
    # dependent parameters
    sigma_ac = ((np.sqrt(m_eff * m_dos**3) * units.k * T) /
                (pi * units.hbar**4 * rho_m * rho_n)).to('s²/kg²')

    # extra multiplication factor for high energies according to Verduin
    # equation (3.126) noticed that A could be balanced out of the equation
    # without branch dependent parameters
    factor_high = (8 * m_dos / (units.k * T)).to('kg/eV')
    # alpha = ((n_BZ + 0.5) * 8 * h_bar_w_BZ / (units.k*T * E_BZ)).to('1/eV')

    def mu(theta):  # see Eq. 3.126
        return (1 - cos(theta)) / 2

    def dcs_lo(mu, E):
        """Phonon cross-section for low energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        two_branch_factor = 1. * (eps_ac_lo**2 / c_s_lo**2) + \
            2. * (eps_ac_tr**2 / c_s_tr**2)
        return (sigma_ac * two_branch_factor /
                (4 * pi * (1 + mu * E / A)**2)).to('cm²')

    def dcs_hi(mu, E):
        """Phonon cross-section for high energies.

        :param E: energy in Joules.
        :param theta: angle in radians."""
        two_branch_factor = \
            1. * ((n_BZ_lo + 0.5) * eps_ac_lo**2 / h_bar_w_BZ_lo) + \
            2. * ((n_BZ_tr + 0.5) * eps_ac_tr**2 / h_bar_w_BZ_tr)
        return ((sigma_ac * factor_high * two_branch_factor * mu * E) /
                (4 * pi * (1 + mu * E / A)**2)).to('cm²')

    def dcs(theta, E):
        m = mu(theta)

        g = interpolate(
            partial(dcs_lo, m), partial(dcs_hi, m),
            h, E_BZ / 4, E_BZ)

        return g(E)

    # should have units of m²/sr
    return dcs


def phonon_cs_fn(s: Settings):
    if s.phonon.model == 'dual':
        return phonon_crosssection_dual_branch(
            s.M_tot, s.rho_m,
            s.phonon.longitudinal.eps_ac,
            s.phonon.longitudinal.c_s,
            s.phonon.longitudinal.alpha,
            s.phonon.transversal.eps_ac,
            s.phonon.transversal.c_s,
            s.phonon.transversal.alpha,
            s.phonon.m_dos,
            s.phonon.m_eff,
            s.phonon.lattice, T=units.T_room,
            interpolate=log_interpolate)
    else:
        return phonon_crosssection(
            s.M_tot, s.rho_m,
            s.phonon.single.eps_ac,
            s.phonon.single.c_s,
            s.phonon.single.alpha,
            s.phonon.m_dos,
            s.phonon.m_eff,
            s.phonon.lattice, T=units.T_room,
            interpolate=log_interpolate)  # , h=lambda x: x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate elastic phonon cross-sections for a material.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in YAML format.")
    args = parser.parse_args()

    s = read_input(args.material_file)
    if 'M_tot' not in s:
        s.M_tot = sum(e.M * e.count for e in s.elements.values())

    E_range = np.logspace(log10(0.01), log10(1000), num=100) * units.eV
    theta_range = np.linspace(0, pi, num=100) * units.rad

    csf = phonon_cs_fn(s)
    cs = DCS(theta_range, E_range[:, None], csf(theta_range, E_range[:, None]),
             x_units='rad', y_units='eV', z_units='cm²', log='y')

    cs.save_gnuplot('{}_phonon.bin'.format(s.name))
