from cstool.parse_input import (read_input, check_settings, cstool_model)
from cstool.phonon import (phonon_cs_fn)
from cslib import (units)

import numpy as np

def test_phonon_cs_fn_single():
    """Tests that the phonon subroutine returns a function that
    can handle arrays and returns correct units."""
    settings = read_input('data/materials/pmma.yaml')
    settings.phonon.model = 'single'
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    fn = phonon_cs_fn(settings)
    W = np.logspace(-2, 3, 100) * units.eV
    theta = np.linspace(0, np.pi, 100) * units.rad
    cs = fn(theta, W[:, None])

    assert cs.shape == (100, 100)
    assert cs.dimensionality == units('m²/sr').dimensionality


def test_phonon_cs_fn_dual():
    """Tests that the phonon subroutine returns a function that
    can handle arrays and returns correct units."""
    settings = read_input('data/materials/pmma.yaml')
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")

    fn = phonon_cs_fn(settings)
    W = np.logspace(-2, 3, 100) * units.eV
    theta = np.linspace(0, np.pi, 100) * units.rad
    cs = fn(theta, W[:, None])

    assert cs.shape == (100, 100)
    assert cs.dimensionality == units('m²/sr').dimensionality
