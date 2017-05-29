from .parse_endf import parse_folder
from .elf import read_elf_data

from cslib import units, Settings

import numpy as np
import os


def _ionization_shells(Z):
    if 'ENDF_DIR' not in os.environ:
        raise EnvironmentError('ENDF_DIR environment variable must be set')
    endf_dir = os.environ['ENDF_DIR']
    if not os.path.isdir(endf_dir):
        raise NotADirectoryError('ENDF_DIR "{}" is not a directory'.format(endf_dir))
    return parse_folder(endf_dir, int(Z))


def ionization_shells(s: Settings):
    shells = []
    for element_name, element in s.elements.items():
        data = _ionization_shells(element.Z)
        for n, shell in sorted(data.items()):
            K, tcs = list(map(list, zip(*shell.cs.data)))
            K = np.array(K)*units.eV
            tcs = np.array(tcs)*units.barn
            tcs *= element.count * shell.occupancy
            shells.append({'E_bind': shell.energy, 'K': K, 'tcs': tcs})
    return shells


def outer_shell_energies(s: Settings):
    elf_data = read_elf_data(s.elf_file)
    osi_energies = []
    for E in elf_data.comments:
        if E < 0 or E > 100:
            break
        osi_energies.append(E)
    return np.array(osi_energies, dtype='f') * units.eV
