import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from cslib import units

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot cross-sections from HDF5 material file.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in HDF5 format.")
    args = parser.parse_args()

    infile = h5.File(args.material_file, 'r')

    properties = {}

    for x in infile['properties']:
        name = x[0].decode('ASCII')
        value = float(x[1]) * units(x[2].decode('ASCII'))
        properties[name] = value

    print('properties:')
    for name, value in properties.items():
        print('{: <16} {}'.format(name, value))
    print()

    def dataset_units(dataset):
        return np.array(dataset) * units(dataset.attrs['units'])

    # elastic
    el_energy_dat = dataset_units(infile['elastic/energy'])
    el_cs_dat = dataset_units(infile['elastic/cross_section'])
    el_icdf_dat = dataset_units(infile['elastic/angle_icdf'])
    print('plotting elastic cross-section...')
    plt.loglog(el_energy_dat.to('eV'), el_cs_dat.to('nm^2'))
    plt.title('elastic cross-section')
    plt.xlabel('$K$ [eV]')
    plt.ylabel('$\sigma$ [m²]')
    plt.show()
    print()

    # inelastic
    inel_energy_dat = dataset_units(infile['inelastic/energy'])
    inel_cs_dat = dataset_units(infile['inelastic/cross_section'])
    inel_icdf_dat = dataset_units(infile['inelastic/w0_icdf'])
    print('plotting inelastic cross-section...')
    plt.loglog(inel_energy_dat.to('eV'), inel_cs_dat.to('nm^2'))
    plt.title('elastic cross-section')
    plt.xlabel('$K$ [eV]')
    plt.ylabel('$\sigma$ [m²]')
    plt.show()
    print()

    # ionization
    ion_energy_dat = dataset_units(infile['ionization/energy'])
    ion_cs_dat = dataset_units(infile['ionization/cross_section'])
    ion_icdf_dat = dataset_units(infile['ionization/dE_icdf'])
    print('plotting ionization cross-section...')
    plt.loglog(ion_energy_dat.to('eV'), ion_cs_dat.to('nm^2'))
    plt.title('ionization cross-section')
    plt.xlabel('$K$ [eV]')
    plt.ylabel('$\sigma$ [m²]')
    plt.show()
    print()

    infile.close()
