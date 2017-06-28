from cstool.parse_input import (read_input, check_settings, cstool_model)
from cstool.inelastic import (inelastic_cs_fn)
from cslib import (units)

import numpy as np

def test_inelastic_cs_fn():
    """Tests that the inelastic subroutine returns a function that
    can handle arrays and returns correct units."""
    settings = read_input('data/materials/pmma.yaml')
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    fn = inelastic_cs_fn(settings)
    K = np.logspace(1, 4, 100) * units.eV
    W = np.logspace(-4, 4, 100) * units.eV
    cs = fn(K, W[:, None])
    print(cs)

    assert cs.shape == (100, 100)
    assert cs.dimensionality == units('m²/eV').dimensionality
