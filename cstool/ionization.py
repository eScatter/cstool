from .parse_endf import parse_zipfiles

from cslib import units, Settings

import numpy as np
import os
import json

from urllib.request import urlopen
from hashlib import sha1
from pkg_resources import resource_string


def loglog_interpolate(x_i, y_i):
    """Interpolates the tabulated values. Linear interpolation
    on a log-log scale.
    Out-of-range behaviour: extrapolation."""

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

        return np.exp(y) * y_i.units

    return f


def obtain_endf_files():
    sources = json.loads(resource_string(__name__, 'data/endf_sources.json').decode("utf-8"))
    os.makedirs('endf.cache', exist_ok=True)
    for name, source in sources.items():
        source['filename'] = 'endf.cache/{}.zip'.format(name)

        if os.path.isfile(source['filename']):
            with open(source['filename'], 'rb') as f:
                if sha1(f.read()).hexdigest() == source['sha1']:
                    print("using cached file {}".format(source['filename']))
                    continue
                else:
                    print("cached file {} has incorrect checksum".format(source['filename']))

        print("downloading {} file".format(name))
        try:
            with urlopen(source['url']) as response:
                data = response.read()
                if sha1(data).hexdigest() != source['sha1']:
                    raise Exception("downloaded file has incorrect checksum")
                with open(source['filename'], 'wb') as f:
                    f.write(data)
        except Exception as e:
            print("failed to download {} file ({})".format(name, e))
            exit()

    return sources


def _ionization_shells(Z):
    sources = obtain_endf_files()
    return parse_zipfiles(sources['atomic_relax']['filename'],
                          sources['electrons']['filename'],
                          int(Z))


def ionization_shells(s: Settings):
    shells = []
    for element_name, element in s.elements.items():
        data = _ionization_shells(element.Z)
        for n, shell in sorted(data.items()):
            K, cs = list(map(list, zip(*shell.cs.data)))
            B = shell.energy
            K = np.array(K)*units.eV
            cs = np.array(cs)*units.barn
            cs *= element.count * shell.occupancy
            shells.append({'B': B, 'K': K, 'cs': cs})
    return shells


def outer_shell_energies(s: Settings):
    osi_energies = s.elf_file.get_outer_shells()
    osi_energies = np.delete(osi_energies.to('eV').magnitude,
        np.where(osi_energies >= 100*units.eV)) * units.eV

    def osi_fun(K):
        # pick the largest energy which is larger then K
        E = np.ndarray(K.shape) * units.eV
        E[:] = np.nan
        for B in sorted(osi_energies):
            E[K > B] = B
        return E

    return osi_fun
