from cslib.dataframe import DataFrame

from itertools import takewhile
import numpy as np


def read_elf_data(filename):
    """ELF files (Energy Loss Function), contain two columns of data,
    starting with undocumented pieces of meta-data.

    This function reads the ELF file and returns a DataFrame object."""
    lines = iter(open(filename, 'r'))

    # extract meta-data
    first_line = next(lines)
    meta_data = list(map(int, first_line.split()))

    # extract data
    parsed_lines = map(lambda l: tuple(map(float, l.split())), lines)
    data = takewhile(lambda v: len(v) > 0 and v[0] > 0, parsed_lines)

    data_array = np.array(list(data),
                          dtype=[('w0', float), ('cs', float)])
    return DataFrame(data_array, units=['eV', ''], comments=meta_data)


if __name__ == "__main__":
    import sys

    elf_data = read_elf_data(sys.argv[1])
    print(elf_data)
