"""
Parses input files of the Cross-section tool, and generates valid input files
from (modified) settings in Python.
"""

from collections import OrderedDict
from ruamel import yaml

from cslib import (
    units)

from cslib.settings import (
    Type, Model, ModelType, Settings, each_value_conforms,
    check_settings, generate_settings, parse_to_model)

from cslib.predicates import (
    predicate,
    is_string, is_integer, file_exists, has_units, is_none, is_)

from .phonon_loss import phonon_loss
from .elf import ELF


def pprint_settings(model, settings):
    dumper = yaml.RoundTripDumper
    dumper.add_representer(ELF, lambda dumper, data : dumper.represent_data(data.filename))
    return yaml.dump(
        generate_settings(settings),
        indent=4, allow_unicode=True, Dumper=dumper)


def quantity(description, unit_str, default=None):
    return Type(description, default=default,
                check=has_units(unit_str),
                generator=lambda v: '{:~P}'.format(v),
                parser=units.parse_expression)


def maybe_quantity(description, unit_str, default=None):
    return Type(description, default=default,
                check=is_none | has_units(unit_str),
                generator=lambda v: v if v is None else '{:~P}'.format(v),
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
    ('single',    ModelType(
        phonon_branch_model, "branch",
        "Only given for single mode, parameters of model.")),
    ('longitudinal', ModelType(
        phonon_branch_model, "branch",
        "Only given for dual mode, parameters of model.")),
    ('transversal', ModelType(
        phonon_branch_model, "branch",
        "Only given for dual mode, parameters of model.")),

    ('energy_loss', maybe_quantity(
        "Phonon loss.", 'eV',
        default=phonon_loss)),

    ('E_BZ',        maybe_quantity(
        "Brioullon zone energy.", 'eV',
        default=lambda s: (units.h**2 / (2*units.m_e * s.lattice**2))
        .to('eV')))])


@predicate("Consistent branch model")
def phonon_check(s: Settings):
    if s.model == 'single' and 'single' in s:
        return True

    if s.model == 'dual' and 'longitudinal' in s and 'transversal' in s:
        return True

    return False

@predicate("Consistent energy diagram")
def energy_check(s: Settings):
    if s.model == 'insulator' or s.model == 'semiconductor':
        if 'band_gap' in s and 'affinity' in s and 'work_func' not in s:
            return True

    if s.model == 'metal':
        if 'band_gap' not in s and 'affinity' not in s and 'work_func' in s:
            return True

    return False

def get_barrier(s: Settings):
    if s.model == 'insulator' or s.model == 'semiconductor':
        if s.fermi > 0*units.eV:
            return s.fermi + s.band_gap/2 + s.affinity
        else:
            return s.band_gap + s.affinity

    if s.model == 'metal':
        return s.fermi + s.work_func

    # It should be impossible to get here, s.model is checked to be insul/semic/metal
    return 0*units.eV

band_structure_model = Model([
    ('model',     Type(
        "Whether the material is of `insulator`, `semiconductor` or `metal` type."
        " Insulators and semiconductors are treated in the same manner",
        check=is_('insulator') | is_('semiconductor') | is_('metal'))),
    ('fermi',     quantity("Fermi energy", 'eV')),
    ('barrier',   quantity("Barrier energy", 'eV', default=get_barrier)),

    # Metals
    ('work_func', maybe_quantity("Work function", 'eV')),

    # Insulators / semiconductors
    ('affinity',  maybe_quantity("Electron affinity", 'eV')),
    ('band_gap',  maybe_quantity("Band gap", 'eV'))
])


cstool_model = Model([
    ('name',      Type("Name of material", default=None,
                       check=is_string)),

    ('rho_m',     quantity("Specific density", 'g/cm³')),

    ('band_structure', ModelType(
        band_structure_model, "band_structure",
        "Band structure of the material. There are two models: metals"
        " and insulators (or semiconductors). Metals need a Fermi energy"
        " and work function, insulators need a Fermi energy, band gap"
        " and affinity. The barrier energy is calculated as Fermi +"
        " work_func in the case of metals, or as Fermi + affinity +"
        " band_gap/2 for insulators.",
        check=energy_check, obligatory=True)),

    ('phonon', ModelType(
        phonon_model, "phonon",
        "We have two choices for modeling phonon scattering: single and"
        " dual branch. The second option is important for crystaline"
        " materials; we then split the scattering in transverse and"
        " longitudinal modes.",
        check=phonon_check, obligatory=True)),

    ('elf_file',  Type(
        "Filename of ELF data (Energy Loss Function). Data can be harvested"
        " from http://henke.lbl.gov/optical_constants/getdb2.html.",
        check=lambda s : True,
        parser=lambda fname : ELF(fname))),

    ('elements',  Type(
        "Dictionary of elements contained in the substance.",
        check=each_value_conforms(element_model, "element"),
        parser=lambda d: OrderedDict((k, parse_to_model(element_model, v))
                                     for k, v in d.items()),
        generator=lambda d: yaml.comments.CommentedMap(
            (k, generate_settings(v))
            for k, v in d.items()))),

    ('M_tot',       maybe_quantity(
        "Total molar mass; this is computed from the `elements` entry.",
        'g/mol',
        default=lambda s: sum(e.M * e.count for e in s.elements.values()))),

    ('rho_n',       maybe_quantity(
        "Number density of atoms or molecules in compound. For instance "
        "in the case of silicon dioxide this is the number density of "
        "groups of two oxygen and one silicon atom, even if SiO2 is not "
        "a molecule per se.", 'cm⁻³',
        default=lambda s: (units.N_A / s.M_tot * s.rho_m).to('cm⁻³')))
])


cstool_model_type = ModelType(
    cstool_model, "cstool",
    """The settings given to cstool should follow a certain hierarchy,
    and each setting is required to have a particular dimensionality.""")


def read_input(filename):
    raw_data = yaml.load(open(filename, 'r', encoding='utf-8'), Loader=yaml.RoundTripLoader)
    settings = parse_to_model(cstool_model, raw_data)
    if not check_settings(settings, cstool_model):
        raise ValueError("Parsed settings do not conform the model.")
    return settings
