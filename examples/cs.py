from noodles.run.run_with_prov import run_parallel_opt
from noodles.display import NCDisplay

from cstool.parse_input import read_input, pprint_settings, cstool_model
from cstool.mott import s_mott_cs
from cstool.phonon import phonon_cs_fn
from cstool.inelastic import inelastic_cs_fn
from cstool.compile import compute_tcs_icdf
from cslib.noodles import registry
from cslib import units
from cslib.dcs import DCS
# from cslib.numeric import log_interpolate

import numpy as np
import h5py as h5

import matplotlib.pyplot as plt


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


def compute_elastic_tcs_icdf(dcs, n):
    print('.', end='', flush=True)

    def integrant(theta):
        return dcs(theta) * 2 * np.pi * np.sin(theta)

    return compute_tcs_icdf(integrant, 0, np.pi, n)


def compute_inelastic_tcs_icdf(dcs, n, K0, K):
    print('.', end='', flush=True)

    def integrant(w):
        return dcs(w)

    return compute_tcs_icdf(integrant, K0, K, n)


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
    pcs = DCS(
        mcs.x.to('rad'),
        e[:, None],
        phonon_cs_fn(s)(mcs.x, e[:, None]).to('cm²'),
        log='y'
    )
    pcs.save_gnuplot('{}_phonon.bin'.format(s.name))

    print("# Merging elastic scattering processes.")

    #@shift(s.fermi.to('eV').magnitude)
    def elastic_cs_fn(a, E):
        return log_interpolate(
            lambda E: phonon_cs_fn(s)(a*units.rad, E*units.eV).to('cm^2').magnitude,
            lambda E: mcs.unsafe(a, E.flat),
            lambda x: x, 100, 200)(E)

    e = np.logspace(-2, 4, 129) * units.eV
    ecs = DCS(
        mcs.x.to('rad'),
        e[:, None],
        elastic_cs_fn(mcs.x.to('rad').magnitude, e.to('eV').magnitude[:, None]) * units('cm^2')
    )
    ecs.save_gnuplot('{}_ecs.bin'.format(s.name))

    outfile = h5.File("{}.mat.hdf5".format(s.name), 'w')

    # elastic
    elastic_grp = outfile.create_group("elastic")
    el_energies = elastic_grp.create_dataset("energy", (129,), dtype='f')
    el_energies[:] = e.magnitude
    el_energies.attrs['units'] = 'eV'
    el_tcs = elastic_grp.create_dataset("cross_section", (129,), dtype='f')
    el_tcs.attrs['units'] = 'nm^2'
    el_icdf = elastic_grp.create_dataset("angle_icdf", (129, 1024), dtype='f')
    el_icdf.attrs['units'] = 'radian'
    print("# Computing elastic total cross-sections and iCDFs.")
    for i, K in enumerate(e):
        def dcs(theta):
            return elastic_cs_fn(theta, K.magnitude)
        tcs, icdf = compute_elastic_tcs_icdf(dcs, 1024)
        el_tcs[i] = tcs/1e-18
        el_icdf[i] = icdf
        #print(el_energies[i], el_tcs[i])
    print()

    #plt.loglog(e, 1/np.array(el_tcs))
    #plt.show()


    e = np.logspace(np.log10(s.fermi.magnitude+0.1), 4, 129) * units.eV

    # inelastic
    inelastic_grp = outfile.create_group("inelastic")
    inel_energies = inelastic_grp.create_dataset("energy", (129,), dtype='f')
    inel_energies[:] = e.magnitude
    inel_energies.attrs['units'] = 'eV'
    inel_tcs = inelastic_grp.create_dataset("cross_section", (129,), dtype='f')
    inel_tcs.attrs['units'] = 'nm^2'
    inel_icdf = inelastic_grp.create_dataset("w0_icdf", (129, 1024), dtype='f')
    inel_icdf.attrs['units'] = 'eV'
    print("# Computing inelastic total cross-sections and iCDFs.")
    for i, K in enumerate(e):
        w0_max = K/2
        if True:
            w0_max = (K - s.fermi)/2

        def dcs(w):
            return inelastic_cs_fn(s)(K, w*units.eV)
        # TODO: use a value for n dependent on K
        tcs, icdf = compute_inelastic_tcs_icdf(dcs, 1024, 1e-4*units.eV, w0_max)
        inel_tcs[i] = tcs/1e-20 # ???
        inel_icdf[i] = icdf
        #print(inel_energies[i], inel_tcs[i])
    print()

    plt.loglog(e, np.array(inel_tcs))
    plt.show()

    outfile.close()
