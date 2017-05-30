from .parse_endf import parse_folder
from .elf import read_elf_data

from cslib import units, Settings

import numpy as np
import os
from warnings import warn


def loglog_interpolate(x_i, y_i):
    """Interpolates the tabulated values. Linear interpolation
    on a log-log scale.
    Out-of-range behaviour: extrapolation if x is too high, and
    zero if x is too low."""

    assert y_i.shape == x_i.shape, "shapes should match"

    x_log_steps = np.log(x_i[1:]/x_i[:-1])
    log_y_i = np.log(y_i.magnitude)

    def f(x):
        x_idx = np.searchsorted(x_i.magnitude.flat,
                                x.to(x_i.units).magnitude.flat)
        mx_idx = np.clip(x_idx - 1, 0, x_i.size - 2)

        # compute the weight factor
        w = np.log(x / np.take(x_i, mx_idx)) \
            / np.take(x_log_steps, mx_idx)

        y = (1 - w) * np.take(log_y_i, mx_idx) \
            + w * np.take(log_y_i, mx_idx + 1)

        if np.any(x_idx == x_i.size):
            warn("Extrapolating ionization cross section above upper bound ({})."
                 .format(x_i.flatten()[-1]))

        # Below the lower bound, return 0. Above the upper bound, extrapolate.
        return (x_idx != 0) * np.exp(y) * y_i.units

    return f


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
