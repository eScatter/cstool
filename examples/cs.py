from noodles.run.run_with_prov import run_parallel_opt
from noodles.display import NCDisplay

from cstool.parse_input import read_input, pprint_settings, cstool_model
from cstool.mott import s_mott_cs
from cstool.phonon import phonon_cs_fn
from cstool.elf import read_elf_data
from cstool.inelastic import inelastic_cs_fn
from cstool.compile import compute_icdf, compute_tcs
from cslib.noodles import registry
from cslib import units
from cslib.dcs import DCS
# from cslib.numeric import log_interpolate

import numpy as np
import h5py as h5
from numpy import (log10)


def log_interpolate(f1, f2, h, a, b):
    """Interpolate two functions `f1` and `f2` using interpolation
    function `h`, which maps [0,1] to [0,1] one-to-one."""
    assert callable(f1)
    assert callable(f2)
    assert callable(h)

    def weight(x):
        return np.clip(np.log(x / a) / np.log(b / a), 0.0, 1.0)

    def g(x):
        w = h(weight(x))
        return (1 - w) * f1(x) + w * f2(x)

    return g


def shift(dE):
    def decorator(cs_fn):
        def shifted(a, E, *args):
            return cs_fn(a, E + dE, *args)
        return shifted
    return decorator


def compute_elastic_tcs(dcs, K, n):
    """
    `set_elastic_data(K, dcs(theta): f -> f)`
    -----------------------------------------
    dcs_int(theta) := dcs(theta) 2 pi sin(theta)
    cumulative_dcs(a) := integrate(0, a, dcs_int)
    tcs = cumulative_dcs(pi) # compute total for normalisation
    elastic_tcs(log(K)) := log(tcs)
    icdf(K, P) -> theta | P == cumulative_dcs(theta)/tcs
    """
    print('.', end='', flush=True)

    def integrant(theta):
        return dcs(theta, K) * 2 * np.pi * np.sin(theta)

    return compute_tcs(integrant, 0, np.pi, n), compute_icdf(integrant, 0, np.pi, n)


if __name__ == "__main__":
    s = read_input("./data/materials/silicon.yaml")

    print(pprint_settings(cstool_model, s))
    print()
    print("Phonon loss: {:~P}".format(s.phonon.energy_loss))
    print("Total molar weight: {:~P}".format(s.M_tot))
    print("Number density: {:~P}".format(s.rho_n))
    print("Brioullon zone energy: {:~P}".format(s.phonon.E_BZ))
    print()
    print("# Computing Mott cross-sections using ELSEPA.")

    e = np.logspace(1, 5, 145) * units.eV
    f_mcs = s_mott_cs(s, e, split=12, mabs=False)

    with NCDisplay() as display:
        mcs = run_parallel_opt(
            f_mcs, n_threads=4, registry=registry,
            jobdb_file='cache.json', display=display)

    mcs.save_gnuplot('{}_mott.bin'.format(s.name))

    print("# Computing Phonon cross-sections.")
    e = np.logspace(-2, 3, 181) * units.eV
    cs = phonon_cs_fn(s)(mcs.x, e[:, None])
    pcs = DCS(mcs.x.to('rad'), e[:, None], cs.to('cm²'), log='y')
    pcs.save_gnuplot('{}_phonon.bin'.format(s.name))

    print("# Merging elastic scattering processes.")

    phonon_cs = phonon_cs_fn(s)

    #@shift(s.fermi.to('eV').magnitude)
    def elastic_cs_fn(a, E):
        return log_interpolate(
            lambda E: phonon_cs(a*units.rad, E*units.eV).to('cm^2').magnitude,
            lambda E: mcs.unsafe(a, E.flat),
            lambda x: x, 100, 200)(E)

    e = np.logspace(-2, 5, 129) * units.eV
    ecs = DCS(mcs.x.to('rad'), e[:, None],
              elastic_cs_fn(
                  mcs.x.to('rad').magnitude,
                  e.to('eV').magnitude[:, None]) * units('cm^2'))
    ecs.save_gnuplot('{}_ecs.bin'.format(s.name))

    outfile = h5.File("{}.mat.hdf5".format(s.name), 'w')
    elastic_grp = outfile.create_group("elastic")
    energies = elastic_grp.create_dataset("energy", (129,), dtype='f')
    energies[:] = e.magnitude
    energies.attrs['units'] = 'eV'
    elastic_tcs = elastic_grp.create_dataset("table", (129, 1024), dtype='f')
    elastic_tcs.attrs['units'] = 'radian'
    elastic_tcs_total = elastic_grp.create_dataset("ecs", (129,), dtype='f')
    elastic_tcs_total.attrs['units'] = 'cm^2'
    print("# Computing elastic total crosssections.")
    for i, K in enumerate(e):
        elastic_tcs_total[i], elastic_tcs[i] = compute_elastic_tcs(
                elastic_cs_fn, K.magnitude, 1024)

    outfile.close()

#    print("Reading inelastic scattering cross-sections.")
#    elf_data = read_elf_data(s.elf_file)
#    K_bounds = (s.fermi + 0.1 * units.eV, 1e4 * units.eV)
#    K = np.logspace(
#        log10(K_bounds[0].to('eV').magnitude),
#        log10(K_bounds[1].to('eV').magnitude), 1024) * units.eV
#    w = np.logspace(
#        log10(elf_data['w0'][0].to('eV').magnitude),
#        log10(K_bounds[1].to('eV').magnitude / 2), 1024) * units.eV
#    ics = DCS.from_function(inelastic_cs_fn(s), K[:, None], w)
#    ics.save_gnuplot('{}_ics.bin'.format(s.name))
