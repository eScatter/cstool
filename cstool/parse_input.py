from cslib import (
    units)
from cslib.settings import (
    Type, Model, Settings, is_settings, each_value_conforms,
    check_settings)
from cslib.predicates import (
    is_string, is_integer, file_exists, has_units, is_none)

from .phonon_loss import phonon_loss

import json
from collections import OrderedDict


def parse_to_model(model, data):
    s = Settings(_model=model)
    for k, v in data.items():
        if k not in model:
            raise KeyError("Key {k} not in model.".format(k=k))
        s[k] = model[k].parser(v)
    return s


def pprint_settings(model, settings):
    return json.dumps(transform_settings(model, settings),
                      indent=4, ensure_ascii=False)


def quantity(description, unit_str, default=None):
    return Type(description, default=default,
                check=has_units(unit_str),
                transformer=lambda v: '{:~P}'.format(v),
                parser=units.parse_expression)


def maybe_quantity(description, unit_str, default=None):
    return Type(description, default=default,
                check=is_none | has_units(unit_str),
                transformer=lambda v: v if v is None else '{:~P}'.format(v),
                parser=lambda s: s if s is None else units.parse_expression(s))

element_model = Model([
    ('count',     Type("Integer abundance", default=None,
                       check=is_integer)),
    ('Z',         Type("Atomic number", default=None,
                       check=is_integer)),
    ('M',         quantity("Molar mass", 'g/mol'))
])


def transform_settings(model, settings):
    return OrderedDict((k, model[k].transformer(v))
                       for k, v in settings.items())

cstool_model = Model([
    ('name',      Type("Name of material", default=None,
                       check=is_string)),

    ('rho_m',     quantity("Specific density", 'g/cm³')),
    ('fermi',     quantity("Fermi energy", 'eV')),
    ('work_func', quantity("Work function", 'eV')),
    ('band_gap',  quantity("Band gap", 'eV')),
    ('lattice',   quantity("Lattice spacing", 'Å')),
    ('c_s',       quantity("Speed of sound", 'km/s')),
    ('eps_ac',    quantity("Accoustic deformation potential", 'eV')),

    ('elf_file',  Type(
        "Filename of ELF data (Energy Loss Function). Data can be harvested"
        " from http://henke.lbl.gov/optical_constants/getdb2.html.",
        check=is_string & file_exists)),

    ('elements',  Type(
        "Dictionary of elements contained in the substance.",
        check=is_settings & each_value_conforms(element_model),
        parser=lambda d: OrderedDict((k, parse_to_model(element_model, v))
                                     for k, v in d.items()),
        transformer=lambda d: OrderedDict((k, transform_settings(element_model, v))
                                          for k, v in d.items()))),

    ('M_tot',       maybe_quantity(
        "Total molar mass; this is computed from the `elements` entry.",
        'g/mol',
        default=lambda s: sum(e.M * e.count for e in s.elements.values()))),

    ('rho_n',       maybe_quantity(
        "Number density of atoms.", 'cm⁻³',
        default=lambda s: (units.N_A / s.M_tot * s.rho_m).to('cm⁻³'))),

    ('phonon_loss', maybe_quantity(
        "Phonon loss.", 'eV',
        default=lambda s: phonon_loss(s.c_s, s.lattice, units.T_room).to('eV'))),

    ('E_BZ',        maybe_quantity(
        "Brioullon zone energy.", 'eV',
        default=lambda s: (units.h**2 / (2*units.m_e * s.lattice**2)).to('eV')))
])


def read_input(filename):
    raw_data = json.load(open(filename, 'r'), object_pairs_hook=OrderedDict)
    settings = parse_to_model(cstool_model, raw_data)
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")
    return settings
