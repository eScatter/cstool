from cslib.dataframe import DataFrame

from itertools import takewhile
import numpy as np


def read_elf_data(filename, print_bool=False):
    """ELF files (Energy Loss Function), contain two columns of data,
    starting with undocumented pieces of meta-data.

    This function reads the ELF file and returns a DataFrame object."""
    lines = iter(open(filename, 'r'))

    # extract meta-data
    first_line = next(lines)
    meta_data = list(map(float, first_line.split()))

    # extract data
    parsed_lines = map(lambda l: tuple(map(float, l.split())), lines)
    data = takewhile(lambda v: len(v) > 0 and v[0] > 0, parsed_lines)

    data_array = np.array(list(data),
                          dtype=[('w0', float), ('elf', float)])
    if print_bool:
        if data_array[0][0] > 0.05:
            print("WARNING: the lowest energy loss ({} eV) tabulated in the ELF data ".format(data_array[0][0]), \
            "is too large. Please add data down to at least 0.05 eV energy losses")
        print("WARNING: the highest energy loss tabulated in the ELF data is", \
        "{} eV. For higher energies, the ELF data is".format(data_array[-1][0]), \
        "extrapolated.")

    return DataFrame(data_array, units=['eV', ''], comments=meta_data)


if __name__ == "__main__":
    import sys

    elf_data = read_elf_data(sys.argv[1])
    print(elf_data)
