from cstool.parse_input import (parse_to_model, check_settings, cstool_model)
from cstool.phonon import (phonon_cs_fn)
from cslib import (units)

import numpy as np

pmma = {
    "name": "pmma",
    "rho_m": "1.192 g/cm³",
    "fermi": "0 eV",
    "work_func": "2.5 eV",

    "phonon_model": {
        "model": "single",
        "lattice": "5.43 Å",
        "single": {
            "c_s": "2750 m/s",
            "eps_ac": "9.2 eV"
            }
        },

    "band_gap": "5.6 eV",
    "elf_file": "data/elf/df_PMMA.dat",
    "elements": {
        "H":  {"count": 8, "Z": 1, "M":  "1.008 g/mol"},
        "C":  {"count": 5, "Z": 6, "M": "12.011 g/mol"},
        "O":  {"count": 2, "Z": 8, "M": "15.999 g/mol"}}
}

settings = parse_to_model(cstool_model, pmma)
if not check_settings(settings, cstool_model):
    raise ValueError("Parsed settings do not conform the model.")


def test_phonon_cs_fn():
    """Tests that the phonon subroutine returns a function that
    can handle arrays and returns correct units."""
    fn = phonon_cs_fn(settings)
    W = np.logspace(-2, 3, 100) * units.eV
    theta = np.linspace(0, np.pi, 100) * units.rad
    cs = fn(W[:, None], theta)

    assert cs.shape == (100, 100)
    assert cs.dimensionality == units('m²/sr').dimensionality
