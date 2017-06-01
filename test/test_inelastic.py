from cstool.parse_input import (parse_to_model, check_settings, cstool_model)
from cstool.inelastic import (inelastic_cs_fn)
from cslib import (units)

import numpy as np

pmma = {
    "name": "pmma",
    "rho_m": "1.192 g/cm³",
    "fermi": "0 eV",
    "work_func": "2.5 eV",

    "phonon": {
        "model": "dual",
        "lattice": "5.43 Å",
        "single": {
            "alpha": "2.13e-7 m²/s",
            "c_s": "2750 m/s",
            "eps_ac": "9.2 eV"},
        "longitudinal": {
            "alpha": "2.00e-7 m²/s",
            "c_s": "2750 m/s",
            "eps_ac": "9.2 eV"},
        "transversal": {
            "alpha": "2.26e-7 m²/s",
            "c_s": "2750 m/s",
            "eps_ac": "9.2 eV"}
        },

    "band_gap": "5.6 eV",
    "elf_file": "data/elf/df_PMMA.dat",
    "elements": {
        "H":  {"count": 8, "Z": 1, "M":  "1.008 g/mol"},
        "C":  {"count": 5, "Z": 6, "M": "12.011 g/mol"},
        "O":  {"count": 2, "Z": 8, "M": "15.999 g/mol"}}
}


def test_inelastic_cs_fn():
    """Tests that the inelastic subroutine returns a function that
    can handle arrays and returns correct units."""
    settings = parse_to_model(cstool_model, pmma)
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    fn = inelastic_cs_fn(settings)
    K = np.logspace(1, 4, 100) * units.eV
    W = np.logspace(-4, 4, 100) * units.eV
    cs = fn(K, W[:, None])
    print(cs)

    assert cs.shape == (100, 100)
    assert cs.dimensionality == units('m²/eV').dimensionality
