from noodles import (schedule)

from cslib import Settings, Q_, DCS
from elsepa.noodles import s_elscata
import numpy as np


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

    return DCS(energies, angles, dcs)


@schedule
def s_sum_dcs(material, **dcs):
    cs = sum(element.count * dcs[symbol].cs
             for symbol, element in material.elements.items())
    first = next(iter(dcs))
    return DCS(dcs[first].energy, dcs[first].angle, cs)


def s_mott_cs(material: Settings, energies):
    """Compute Mott crosssections.

    Runs `elscata` for each element in the material. Then adds
    the cross-sections proportional to the composition of the
    material."""
    def s_atomic_dcs(Z):
        no_muffin_Z = [1, 7, 8]

        settings = Settings(
            IZ=Z, MNUCL=3, MELEC=4, IELEC=-1, MEXCH=1, IHEF=0,
            MCPOL=2, MABS=0, MUFFIN=0 if Z in no_muffin_Z else 1,
            EV=energies)

        f_result = s_elscata(settings)
        return s_get_dcs(f_result, energies)
   
    dcs = {symbol: s_atomic_dcs(element.Z)
           for symbol, element in material.elements.items()}
    return s_sum_dcs(material, **dcs)
