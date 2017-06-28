from noodles.run.run_with_prov import run_parallel_opt
from noodles.display import NCDisplay

from cstool.parse_input import read_input, pprint_settings, cstool_model
from cstool.mott import s_mott_cs
from cstool.phonon import phonon_cs_fn
from cstool.inelastic import inelastic_cs_fn
from cstool.compile import compute_tcs_icdf
from cstool.ionization import ionization_shells, outer_shell_energies, \
                              loglog_interpolate as ion_loglog_interp
from cslib.noodles import registry
from cslib import units
from cslib.dcs import DCS
from cslib.numeric import log_interpolate

import numpy as np
import h5py as h5


def compute_elastic_tcs_icdf(dcs, P):
    def integrant(theta):
        return dcs(theta) * 2 * np.pi * np.sin(theta)

    return compute_tcs_icdf(integrant, 0*units('rad'), np.pi*units('rad'), P)


def compute_inelastic_tcs_icdf(dcs, P, K0, K, max_interval):
    def integrant(w):
        return dcs(w)

    return compute_tcs_icdf(integrant, K0, K, P,
        sampling = np.min([100000,
            int(np.ceil((K - K0) / max_interval))
            ]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Create HDF5 file from material definition.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in YAML format.")
    args = parser.parse_args()

    s = read_input(args.material_file)

    print(pprint_settings(cstool_model, s))
    print()
    print("Phonon loss: {:~P}".format(s.phonon.energy_loss))
    print("Total molar weight: {:~P}".format(s.M_tot))
    print("Number density: {:~P}".format(s.rho_n))
    print("Brillouin zone energy: {:~P}".format(s.phonon.E_BZ))
    print("Barrier energy: {:~P}".format(s.band_structure.barrier))
    print()
    print("# Computing Mott cross-sections using ELSEPA.")

    e_mcs = np.logspace(1, 5, 145) * units.eV
    f_mcs = s_mott_cs(s, e_mcs, split=12, mabs=False)

    with NCDisplay() as display:
        mcs = run_parallel_opt(
            f_mcs, n_threads=4, registry=registry,
            jobdb_file='cache.json', display=display)

    print("# Merging elastic scattering processes.")

    def elastic_cs_fn(a, E):
        return log_interpolate(
            lambda E: phonon_cs_fn(s)(a, E).to('cm^2').magnitude,
            lambda E: mcs.unsafe(a, E.to('eV').magnitude.flat),
            lambda x: x, 100*units.eV, 200*units.eV
        )(E)*units('cm^2/rad')

    properties = {
        'fermi': (s.band_structure.fermi, 'eV'),
        'barrier': (s.band_structure.barrier, 'eV'),
        'phonon_loss': (s.phonon.energy_loss, 'eV'),
        'density': (s.rho_n, 'm^-3'),
        'effective_A': (sum(e.M * e.count for e in s.elements.values())/(units.N_A*sum(e.count for e in s.elements.values())), 'g')
    }
    if s.band_structure.model == 'insulator' or s.band_structure.model == 'semiconductor':
        properties['band_gap'] = (s.band_structure.band_gap, 'eV')

    # write output
    outfile = h5.File("{}.mat.hdf5".format(s.name), 'w')

    outfile.attrs['name'] = s.name
    outfile.attrs['conductor_type'] = s.band_structure.model

    hdf_properties = outfile.create_dataset(
        "properties", (len(properties),), dtype=np.dtype([
            ('name', h5.special_dtype(vlen=bytes)),
            ('value', float),
            ('unit', h5.special_dtype(vlen=bytes))
        ]))
    for i, (name, value) in enumerate(properties.items()):
        hdf_properties[i] = (name, value[0].to(value[1]).magnitude, value[1])

    # elastic
    e_el = np.logspace(-2, 4, 129) * units.eV
    p_el = np.linspace(0.0, 1.0, 1024)

    elastic_grp = outfile.create_group("elastic")
    el_energies = elastic_grp.create_dataset("energy", data=e_el.to('eV'))
    el_energies.attrs['units'] = 'eV'
    el_tcs = elastic_grp.create_dataset("cross_section", e_el.shape)
    el_tcs.attrs['units'] = 'm^2'
    el_icdf = elastic_grp.create_dataset("angle_icdf", (e_el.shape[0], p_el.shape[0]))
    el_icdf.attrs['units'] = 'radian'
    print("# Computing elastic total cross-sections and iCDFs.")
    for i, K in enumerate(e_el):
        def dcs(theta):
            return elastic_cs_fn(theta, K)
        tcs, icdf = compute_elastic_tcs_icdf(dcs, p_el)
        el_tcs[i] = tcs.to('m^2')
        el_icdf[i] = icdf.to('rad')
        print('.', end='', flush=True)
    print()

    # inelastic
    e_inel = np.logspace(np.log10(s.band_structure.fermi.magnitude+0.1), 4, 129) * units.eV
    p_inel = np.linspace(0.0, 1.0, 1024)

    inelastic_grp = outfile.create_group("inelastic")
    inel_energies = inelastic_grp.create_dataset("energy", data=e_inel.to('eV'))
    inel_energies.attrs['units'] = 'eV'
    inel_tcs = inelastic_grp.create_dataset("cross_section", e_inel.shape)
    inel_tcs.attrs['units'] = 'm^2'
    inel_icdf = inelastic_grp.create_dataset("w0_icdf", (e_inel.shape[0], p_inel.shape[0]))
    inel_icdf.attrs['units'] = 'eV'
    print("# Computing inelastic total cross-sections and iCDFs.")
    for i, K in enumerate(e_inel):
        w0_max = K-s.band_structure.fermi # it is not possible to lose so much energy that the
        # primary electron ends up below the Fermi level in an inelastic
        # scattering event

        def dcs(w):
            return inelastic_cs_fn(s)(K, w)
        tcs, icdf = compute_inelastic_tcs_icdf(dcs, p_inel,
            s.elf_file.get_min_energy(), w0_max,
            s.elf_file.get_min_energy_interval())
        inel_tcs[i] = tcs.to('m^2')
        inel_icdf[i] = icdf.to('eV')
        print('.', end='', flush=True)
    print()

    # ionization
    e_ion = np.logspace(0, 4, 1024) * units.eV
    p_ion = np.linspace(0.0, 1.0, 1024)

    print("# Computing ionization energy probabilities")
    shells = ionization_shells(s)

    tcstot_at_K = np.zeros(e_ion.shape) * units('m^2')
    for shell in reversed(shells):
        shell['cs_at_K'] = np.zeros(e_ion.shape) * units('m^2')
        margin = 10*units.eV
        i_able = ((e_ion+margin) > shell['B'])
        j_able = (shell['K'] > shell['B']) & (shell['cs'] > 0*units('m^2'))
        shell['cs_at_K'][i_able] = ion_loglog_interp(
            shell['K'][j_able], shell['cs'][j_able])((e_ion+margin)[i_able]).to('m^2')
        tcstot_at_K += shell['cs_at_K']

    Pcum_at_K = np.zeros(e_ion.shape)
    for shell in reversed(shells):
        shell['P_at_K'] = np.zeros(e_ion.shape)
        i_able = (tcstot_at_K > 0*units('m^2'))
        shell['P_at_K'][i_able] = shell['cs_at_K'][i_able]/tcstot_at_K[i_able]
        Pcum_at_K += shell['P_at_K']
        shell['Pcum_at_K'] = np.copy(Pcum_at_K)

    ionization_icdf = np.ndarray((e_ion.shape[0], p_ion.shape[0]))*units.eV
    for j, P in enumerate(p_ion):
        icdf_at_P = np.ndarray(e_ion.shape) * units.eV
        icdf_at_P[:] = np.nan
        for shell in shells:
            icdf_at_P[P < shell['Pcum_at_K']] = shell['B']
        icdf_at_P[e_ion < 100*units.eV] = np.nan
        icdf_at_P[icdf_at_P < 50*units.eV] = np.nan
        icdf_at_P[np.isnan(icdf_at_P)] = outer_shell_energies(s)(e_ion)[np.isnan(icdf_at_P)]
        ionization_icdf[:, j] = icdf_at_P

    # write ionization
    ionization_grp = outfile.create_group("ionization")
    ion_energies = ionization_grp.create_dataset("energy", data=e_ion.to('eV'))
    ion_energies.attrs['units'] = 'eV'

    ion_icdf = ionization_grp.create_dataset("dE_icdf", data=ionization_icdf.to('eV'))
    ion_icdf.attrs['units'] = 'eV'

    ionization_osi = ionization_grp.create_dataset("outer_shells", data=s.elf_file.get_outer_shells().to('eV'))
    ionization_osi.attrs['units'] = 'eV'

    outfile.close()
