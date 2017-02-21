from cslib import (
    units)

from cslib.settings import (
    Type, Model, Settings, is_settings, each_value_conforms,
    check_settings, conforms)

from cslib.predicates import (
    Predicate,
    is_string, is_integer, file_exists, has_units, is_none, is_)

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

phonon_branch_model = Model([
    ('alpha',     maybe_quantity(
        "Bending in dispersion relation. (TV Eq. 3.112)",
        'm²/s', default=units('0 m²/s'))),
    ('eps_ac',    quantity("Accoustic deformation potential", 'eV')),
    ('c_s',       quantity("Speed of sound", 'km/s'))])

phonon_model = Model([
    ('model',     Type(
        "Whether the model is the `single` or `dual` mode.",
        check=is_('single') | is_('dual'),
        default="single")),
    ('m_eff',     maybe_quantity(
        "Effective mass.", 'g', default=units('1 m_e'))),
    ('m_dos',     maybe_quantity(
        "Density of state mass.", 'g', default=units('1 m_e'))),
    ('lattice',   quantity("Lattice spacing", 'Å')),
    ('single',    Type(
        "Only given for single mode, parameters of model.",
        check=is_none | conforms(phonon_branch_model),
        parser=lambda d: parse_to_model(phonon_branch_model, d),
        transformer=lambda d: transform_settings(phonon_branch_model, d))),
    ('longitudinal', Type(
        "Only given for dual mode, parameters of model.",
        check=is_none | conforms(phonon_branch_model),
        parser=lambda d: parse_to_model(phonon_branch_model, d),
        transformer=lambda d: transform_settings(phonon_branch_model, d))),
    ('transversal', Type(
        "Only given for dual mode, parameters of model.",
        check=is_none | conforms(phonon_branch_model),
        parser=lambda d: parse_to_model(phonon_branch_model, d),
        transformer=lambda d: transform_settings(phonon_branch_model, d))),

    ('phonon_loss', maybe_quantity(
        "Phonon loss.", 'eV',
        default=lambda s: phonon_loss(s.c_s, s.lattice, units.T_room)
        .to('eV'))),

    ('E_BZ',        maybe_quantity(
        "Brioullon zone energy.", 'eV',
        default=lambda s: (units.h**2 / (2*units.m_e * s.lattice**2))
        .to('eV')))])


@Predicate
def phonon_check(s: Settings):
    if not check_settings(s, phonon_model):
        return False

    if s.model == 'single' and 'single' in s:
        return True

    if s.model == 'dual' and 'longitudinal' in s and 'transversal' in s:
        return True

    return False


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

    ('phonon_model', Type(
        "We have two choices for modeling phonon scattering: single and"
        " dual branch. The second option is important for crystaline"
        " materials; we then split the scattering in transverse and"
        " longitudinal modes.",
        check=phonon_check, obligatory=True,
        parser=lambda d: parse_to_model(phonon_model, d),
        transformer=lambda d: transform_settings(phonon_model, d))),

    ('elf_file',  Type(
        "Filename of ELF data (Energy Loss Function). Data can be harvested"
        " from http://henke.lbl.gov/optical_constants/getdb2.html.",
        check=is_string & file_exists)),

    ('elements',  Type(
        "Dictionary of elements contained in the substance.",
        check=is_settings & each_value_conforms(element_model),
        parser=lambda d: OrderedDict((k, parse_to_model(element_model, v))
                                     for k, v in d.items()),
        transformer=lambda d: OrderedDict(
            (k, transform_settings(element_model, v))
            for k, v in d.items()))),

    ('M_tot',       maybe_quantity(
        "Total molar mass; this is computed from the `elements` entry.",
        'g/mol',
        default=lambda s: sum(e.M * e.count for e in s.elements.values()))),

    ('rho_n',       maybe_quantity(
        "Number density of atoms.", 'cm⁻³',
        default=lambda s: (units.N_A / s.M_tot * s.rho_m).to('cm⁻³')))
])


def read_input(filename):
    raw_data = json.load(open(filename, 'r'), object_pairs_hook=OrderedDict)
    settings = parse_to_model(cstool_model, raw_data)
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")
    return settings
