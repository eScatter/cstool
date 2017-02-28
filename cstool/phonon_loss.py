from cslib import units
from scipy.integrate import quad

import numpy as np


def branch_loss(c_s, alpha, lattice, T):
    """Compute the net average energy loss of phonons.

    :param c_s_lo: speed of sound for longitudinal mode (m/s)
    :param alpha_lo: relates to the bending of the dispersion relation
    towards the Brillouin zone boundary for longitudinal mode (m²/s)
    :param lattice: Lattice spacing (A)
    :param T: Temperature (K)
    """

    kT = (1 * units.k * T).to('J')  # 'k' is Boltzmann constant

    # Wave factor at 1st Brillouin Zone Boundary
    k_BZ = 2 * pi / lattice

    # Calculate average net loss per acoustic scattering event

    # Isotropic Dispersion Relation, Verduin (Eq. 3.112)
    def h_bar_w_AC(k, c_s, alpha):
        return units.hbar * (c_s * k - alpha * k**2).to('1/s')  # Verduin Eq. 3.114

    # Bose-Einstein distribution, Verduin (Eq. 3.117)
    def N_BE(k, c_s, alpha):
        return 1. / np.expm1(h_bar_w_AC(k, c_s, alpha) / kT)

    # (Verduin Eq. 3.116)
    def nominator(k, c_s, alpha):
        return  h_bar_w_AC(k, c_s, alpha) * k**2

    def denominator(k, c_s, alpha):
        return (2 * N_BE(k, c_s, alpha) + 1) * k**2

    # TO DO: strip the units https://pint.readthedocs.io/en/0.7.2/wrapping.html
    y1, err1 = quad(nominator, 0, k_BZ, args=(c_s, alpha,))
    y2, err2 = quad(denominator, 0, k_BZ, args=(c_s, alpha,))
    # TO DO: assign the units back

    return  (y1 / y2).to('J')


def phonon_loss(s: Settings):
    float energy_loss = 0
    if s.phonon.model == 'dual':
        # Calculate net average energy loss of multiple branches:
        # 1*Longitidunal + 2*Transversal Branch
        energy_loss = (1.0 * branch_loss(s.phonon.longitudinal.c_s,
                                         s.phonon.longitudinal.alpha,
                                         s.phonon.lattice,
                                         T=units.T_room) +
                       2.0 * branch_loss(s.phonon.transversal.c_s,
                                         s.phonon.transversal.alpha,
                                         s.phonon.lattice,
                                         T=units.T_room)) / 3
        return energy_loss
    else:
        # Calculate net average energy loss of single branch:
        # 1*Longitidunal branch
        energy_loss = 1.0 * branch_loss(s.phonon.single.c_s,
                                        s.phonon.single.alpha,
                                        s.phonon.lattice,
                                        T=units.T_room)
        return energy_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate energy loss for AC phonons for a material.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in JSON format.")
    args = parser.parse_args()

    s = read_input(args.material_file)

    # def phonon_loss(c_s, lattice, T):
    float energy_loss = phonon_loss(s)
    print("Phonon energy loss:" + str(energy_loss) + "eV for")
    print "For" s.phonon.model, "branch"
