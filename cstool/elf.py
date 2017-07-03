from itertools import takewhile
from warnings import warn
import numpy as np
from cslib import units

class ELF:
    """Handles ELF (Energy Loss Function) files, which consist of two
    columns of data: photon energy in eV and the ELF, Im[-1/ε].

    The first line contains a list of outer-shell binding energies
    terminated by "-1". The last line is "-1 -1".

    This class reads such a file and exposes the data."""

    def __init__(self, filename):
        self.filename = filename
        self.__elf_x, self.__elf_y, osi = self.__read(filename)
        self.__elf_fn = self.__loglog_interpolate(self.__elf_x, self.__elf_y)
        self.__osi = osi

        if self.__elf_x[0] > 0.05 * units.eV:
            warn("The lowest energy loss ({}) tabulated in the ELF data "
                 "is too large. Please add data down to at least 0.05 eV "
                 "energy losses".format(self.__elf_x[0]))

    def get_elf(self, omega):
        return self.__elf_fn(omega)

    def get_outer_shells(self):
        return np.copy(self.__osi.to('eV').magnitude) * units.eV

    def get_min_energy(self):
        return self.__elf_x[0]
    def get_max_energy(self):
        return self.__elf_x[-1]

    def get_min_energy_interval(self):
        intervals = self.__elf_x[1:] - self.__elf_x[:-1]
        return np.min(intervals)

    def __call__(self, x):
        return self.get_elf(x)

    def __read(self, filename):
        lines = iter(open(filename, 'r'))

        # extract meta-data
        first_line = next(lines)
        outer_shells = np.array(list(map(float, first_line.split())))
        outer_shells = np.delete(outer_shells, np.where(outer_shells <= 0)) * units.eV

        # extract data
        parsed_lines = map(lambda l: tuple(map(float, l.split())), lines)
        data = takewhile(lambda v: len(v) > 0 and v[0] > 0, parsed_lines)
        data_array = np.array(list(data))

        elf_x = data_array[:,0] * units.eV
        elf_y = data_array[:,1] * units.dimensionless

        return elf_x, elf_y, outer_shells

    def __loglog_interpolate(self, x_i, y_i):
        """Interpolates the tabulated values. Linear interpolation
        on a log-log scale.
        Out-of-range behaviour: extrapolation if x is too high, and
        zero if x is too low."""

        assert y_i.shape == x_i.shape, "shapes should match"

        x_log_steps = np.log(x_i[1:]/x_i[:-1])
        log_y_i = np.log(y_i.magnitude)

        def f(x):
            x_idx = np.searchsorted(x_i.magnitude,
                                    x.to(x_i.units).magnitude)
            mx_idx = np.clip(x_idx - 1, 0, x_i.size - 2)

            # compute the weight factor.
            # Have to strip the units here, because pint does not like "where".
            w = np.log((x / np.take(x_i, mx_idx)).magnitude, where=x>0*x.units, out=np.zeros(x.shape)) \
                / np.take(x_log_steps, mx_idx)

            y = (1 - w) * np.take(log_y_i, mx_idx) \
                + w * np.take(log_y_i, mx_idx + 1)

            if np.any(x_idx == x_i.size):
                warn("Extrapolating ELF data above upper bound ({})."
                     .format(x_i.flatten()[-1]))

            # Below the lower bound, return 0. Above the upper bound, extrapolate.
            # For high energy, a power law is expected, but for low
            # energies we don't know anything, so we want no energy loss.
            return (x_idx != 0) * np.exp(y) * y_i.units

        return f
