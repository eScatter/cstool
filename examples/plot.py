import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from collections import defaultdict
import math

from cslib import units

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot cross-sections from HDF5 material file.')
    parser.add_argument('material_file', type=str,
        help="Filename of material in HDF5 format.")
    parser.add_argument('--elastic', action='store_true')
    parser.add_argument('--inelastic', action='store_true')
    parser.add_argument('--ionization', action='store_true')
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

    if not (args.elastic or args.inelastic or args.ionization):
        print('Not plotting anything. Use the -h flag for usage.')

    def dataset_units(dataset):
        return np.array(dataset) * units(dataset.attrs['units'])

    if args.elastic:
        print('loading elastic data...')
        el_energy_dat = dataset_units(infile['elastic/energy'])
        el_cs_dat = dataset_units(infile['elastic/cross_section'])
        el_icdf_dat = dataset_units(infile['elastic/angle_icdf'])

        plt.loglog(el_energy_dat.to('eV'), el_cs_dat.to('nm^2'))
        plt.title('elastic cross-section')
        plt.xlabel('$K$ [eV]')
        plt.ylabel('$\sigma$ [nm²]')
        plt.show()

    if args.inelastic:
        print('loading inelastic data...')
        inel_energy_dat = dataset_units(infile['inelastic/energy'])
        inel_cs_dat = dataset_units(infile['inelastic/cross_section'])
        inel_icdf_dat = dataset_units(infile['inelastic/w0_icdf'])

        plt.loglog(inel_energy_dat.to('eV'), inel_cs_dat.to('nm^2'))
        plt.title('inelastic cross-section')
        plt.xlabel('$K$ [eV]')
        plt.ylabel('$\sigma$ [nm²]')
        plt.show()

    if args.ionization:
        print('loading ionization data...')
        ion_energy_dat = dataset_units(infile['ionization/energy'])
        ion_icdf_dat = dataset_units(infile['ionization/dE_icdf'])

        shells = {}
        for i, K in enumerate(ion_energy_dat):
            shell_p = defaultdict(int)
            for B in ion_icdf_dat[i, :]:
                if not math.isnan(B.to('eV').magnitude):
                    shell_p[B.to('eV').magnitude] += 1
            for B, P in shell_p.items():
                if B not in shells:
                    shells[B] = {}
                shells[B][K.to('eV').magnitude] = P/ion_icdf_dat.shape[1]

        legends = []
        for B, K_P in shells.items():
            K, P = zip(*K_P.items())
            legends.append("$B = {}$ eV".format(B))
            plt.semilogx(K, P)
        plt.legend(legends)
        plt.title('ionization energy probability')
        plt.xlabel('$K$ [eV]')
        plt.ylabel('P')
        plt.show()

    infile.close()
