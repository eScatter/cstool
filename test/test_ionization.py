from cstool.parse_input import (read_input, check_settings, cstool_model)
from cstool.ionization import (ionization_shells, outer_shell_energies)
from cslib import (units)

import numpy as np


def test_ionization_shells():
    """Tests that the ionization shells have the correct shape and units."""
    settings = read_input('data/materials/pmma.yaml')
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    shells = ionization_shells(settings)

    assert len(shells) > 0
    for shell in shells:
        assert shell['B'].dimensionality == units('eV').dimensionality
        assert shell['K'].dimensionality == units('eV').dimensionality
        assert shell['cs'].dimensionality == units('m^2').dimensionality


def test_outer_shell_energies():
    """Tests that the outer shell energies have the correct units."""
    settings = read_input('data/materials/pmma.yaml')
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    fn = outer_shell_energies(settings)
    K = np.logspace(1, 4, 100) * units.eV
    osi = fn(K)

    assert osi.shape == (100,)
    assert osi.dimensionality == units('eV').dimensionality
