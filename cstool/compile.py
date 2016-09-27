"""
Creates differential cross-sections from several sources, integrates them into
distribution functions (CDF⁻¹), and writes them in a format that the CUDA low
energy electron scattering simulator understands.

The C++ code does the following
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a material object `mat` with fields
    `name`, `fermi`, `fermi+work_func`, `phonon_loss`, `number_density`.
2. If a band-gap is given, also include `band_gap`.

3. Read elastic cross-sections: set them in the material.

4. Read inelastic cross-sections: set them in the material.

5. Read ionization cross-sections: set them in the material with a twist:
    B = binding_energy
    values_plus_B = {}
    for each table in TCS: -> (first: double, second: TCS)
        values_plus_B[first + B] = second
    set values_plus_B in material
6. This adds the binding-energy that was subtracted from the same sample in
   the Python script. We use absolute values anyway, so we can skip this.

7. Read the ELF file (now called `outer-shell`) again and read the first line
   multiplying thevalues with the elementary charge (or 1 eV), somehow only
   including values that are less than 100 eV.
   These are put in an array and given as `outer_shell_ionization_data`.

8. Some statistics are shown which we already know.
   The entire material is serialised.

Serialisation
~~~~~~~~~~~~~

The material has the following fields:
    string  name;
    float   fermi, barrier;
    optional<float> band_gap;
    float   phonon_loss, density (which is number_density);

    map<f,f> elastic_tcs
    map<f,map<f,f>> elastic_icdf
    map<f,f> inelastic_tcs
    map<f,map<f,f>> inelastic_icdf
    map<f,map<f,f>> ionization_tcs
    vector<f> osi_energies

Serialisation is done through the `archive` class in `common/archive.hh`.
In the future this class needs to be replaced with HDF5 format.

Each `put_*` call first writes a single char representing the datum type.
This is done by an enum that we represent here as `type_id`.
If the data is atomic, the following sizeof(type) bytes are the actual data.

* `put_string` gives the type_id belonging to 'string' followed by a uint64
  containing the byte-size of the string, followed by the character data.

* `put_blob` gives the type_id belonging to 'blob' followed by a uint64
  containing the byte-size of the blob, followed by the blob data.

* In the `material.cc` some helper functions have been defined:

    - `put_vector<double>` writes uint32 with length of vector, then N
      times a `put_float64`.
    - `put_map<double, double>` writes uint32 with size of map, then
      N*2 floats64.
    - `put_nested_map<double, map>` writes uint32 with size of map, then
      N times first a float64, then a map<double, double>.

Given this structure the `.mat` file format looks like:
    string          name
    f64             fermi
    f64             barrier
    bool            has_band_gap
    ?<f64>          band_gap
    f64             phonon_loss
    f64             number_density
    map<f,f>        elastic_tcs
    map<f,map<f,f>  elastic_icdf
    map<f,f>        inelastic_tcs
    map<f,map<f,f>  inelastic_icdf
    map<f,map<f,f>  ionization_tcs
    vec<f>          osi_energies
"""

from collections import OrderedDict

type_id_lst = ['bool', 'int8', 'uint8', 'int16', 'uint16',
               'int32', 'uint32', 'int64', 'uint64',
               'float32', 'float64', 'string', 'blob']

type_id = OrderedDict(zip(type_id_lst, range(len(type_id_lst))))

print(type_id)
