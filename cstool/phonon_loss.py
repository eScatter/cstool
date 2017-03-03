from cslib import units

import numpy as np

from scipy.integrate import quad


def branch_loss(c_s, alpha, lattice, T):
    """Compute the net average energy loss of phonons.

    :param c_s_lo: speed of sound for longitudinal mode (m/s)
    :param alpha_lo: relates to the bending of the dispersion relation
    towards the Brillouin zone boundary for longitudinal mode (m²/s)
    :param lattice: Lattice spacing (A)
    :param T: Temperature (K)
    """
    # Wave factor at 1st Brillouin Zone Boundary
    k_BZ = 2 * np.pi / lattice

    h_bar_m = units('1 hbar').to('J s').magnitude
    k_BZ_m = k_BZ.to('1/m').magnitude
    c_s_m = c_s.to('m/s').magnitude
    alpha_m = alpha.to('m^2/s').magnitude
    kT_m = (1 * units.k * T).to('J').magnitude

    # Calculate average net loss per acoustic scattering event
    # Isotropic Dispersion Relation, Verduin (Eq. 3.112)
    def h_bar_w_AC(k):
        return h_bar_m * (c_s_m * k - alpha_m * k**2)  # Verduin Eq. 3.114

    # Bose-Einstein distribution, Verduin (Eq. 3.117)
    def N_BE(k):
        return 1. / np.expm1(h_bar_w_AC(k) / kT_m)

    # (Verduin Eq. 3.116)
    def nominator(k):
        return h_bar_w_AC(k) * k**2

    def denominator(k):
        return (2 * N_BE(k) + 1) * k**2

    # TO DO: strip the units https://pint.readthedocs.io/en/0.7.2/wrapping.html

    y1, err1 = quad(nominator, 0, k_BZ_m)
    y2, err2 = quad(denominator, 0, k_BZ_m)
    # TO DO: assign the units back

    return ((y1 / y2) * units('J')).to('eV')


def phonon_loss(phonon):
    if phonon.model == 'dual':
        # Calculate net average energy loss of multiple branches:
        # 1*Longitidunal + 2*Transversal Branch
        energy_loss = (1.0 * branch_loss(phonon.longitudinal.c_s,
                                         phonon.longitudinal.alpha,
                                         phonon.lattice,
                                         T=units.T_room) +
                       2.0 * branch_loss(phonon.transversal.c_s,
                                         phonon.transversal.alpha,
                                         phonon.lattice,
                                         T=units.T_room)) / 3
        return energy_loss
    else:
        # Calculate net average energy loss of single branch:
        # 1*Longitidunal branch
        energy_loss = 1.0 * branch_loss(phonon.single.c_s,
                                        phonon.single.alpha,
                                        phonon.lattice,
                                        T=units.T_room)
        return energy_loss


if __name__ == "__main__":
    import argparse
    from .parse_input import read_input

    parser = argparse.ArgumentParser(
        description='Calculate energy loss for AC phonons for a material.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in JSON format.")
    args = parser.parse_args()

    s = read_input(args.material_file)

    # def phonon_loss(c_s, lattice, T):
    energy_loss = phonon_loss(s.phonon)
    print("Phonon energy loss: {:~P}".format(energy_loss.to('eV')))
    print("For " + s.phonon.model + " branch")
