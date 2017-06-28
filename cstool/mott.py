from noodles import (schedule, schedule_hint)

from cslib import Settings, Q_
from cslib.dcs import DCS
from elsepa import elscata
import numpy as np


@schedule_hint(
    display="Running 'elscata' for Z={s.IZ},"
            " K=[{s.EV[0]:.2e~P}, ... ({s.EV.size})]  ",
    store=True,
    confirm=True,
    version="0.1.0")
def s_elscata(s):
    return elscata(s)


@schedule
def s_get_dcs(result, energies):
    def mangle_energy(e):
        s = 'dcs_{:.4e}'.format(e.to('eV').magnitude) \
            .replace('.', 'p').replace('+', '')
        return s[:9] + s[10:]

    dcs_keys = [mangle_energy(e) for e in energies]
    angles = result[dcs_keys[0]]['THETA']
    dcs = np.array([result[k]['DCS[0]'].to('cm²/sr').magnitude
                   for k in dcs_keys]) * Q_('cm²/sr')

    return DCS(angles, energies[:, None], dcs,
               x_units='rad', y_units='eV', log='y')


@schedule
def s_join_dcs(*dcs_lst):
    angle = dcs_lst[0].x
    energy = np.concatenate(
        [dcs.y for dcs in dcs_lst], axis=0) \
        * dcs_lst[0].y.units
    cs = np.concatenate(
        [dcs.z for dcs in dcs_lst], axis=0) \
        * dcs_lst[0].z.units
    return DCS(angle, energy, cs, log='y')


@schedule
def s_sum_dcs(material, **dcs):
    cs = sum(element.count * dcs[symbol].z
             for symbol, element in material.elements.items())
    first = next(iter(dcs))
    return DCS(dcs[first].x, dcs[first].y, cs, log='y')


def s_mott_cs(material: Settings, energies, split=4, mabs=False):
    """Compute Mott crosssections.

    Runs `elscata` for each element in the material. Then adds
    the cross-sections proportional to the composition of the
    material.

    :param material:
        Settings object containing parameters for this material.
    :param energies:
        Array quantity, 1-d array with dimension of energy.
    :param mabs:
        Enable absorbtion potential (computations can take longer).
    :param split:
        Elsepa can take a long time to compute. This splits the
        `energies` array in `split` parts, creating as many jobs
        that may run in parallel.
    """
    def split_array(a, n):
        m = np.arange(0, a.size, a.size/n)[1:].astype(int)
        return np.split(a, m)

    chunks = split_array(energies, split)

    def s_atomic_dcs(Z):
        no_muffin_Z = [1, 7, 8]

        settings = [Settings(
            IZ=Z, MNUCL=3, MELEC=4, IELEC=-1, MEXCH=1, IHEF=0,
            MCPOL=2, MABS=1 if mabs else 0,
            MUFFIN=0 if Z in no_muffin_Z else 1,
            EV=e) for e in chunks]

        f_results = [s_get_dcs(s_elscata(s), s.EV) for s in settings]

        return s_join_dcs(*f_results)

    dcs = {symbol: s_atomic_dcs(element.Z)
           for symbol, element in material.elements.items()}
    return s_sum_dcs(material, **dcs)
