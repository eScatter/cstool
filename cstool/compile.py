"""
Creates differential cross-sections from several sources, integrates them into
distribution functions (CDF⁻¹), and writes them in a format that the CUDA low
energy electron scattering simulator understands.

The C++ code does the following
===============================

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
   This adds the binding-energy that was subtracted from the same sample in
   the Python script. We use absolute values anyway, so we can skip this.

7. Read the ELF file (now called `outer-shell`) again and read the first line
   multiplying thevalues with the elementary charge (or 1 eV), somehow only
   including values that are less than 100 eV.
   These are put in an array and given as `outer_shell_ionization_data`.

8. Some statistics are shown which we already know.
   The entire material is serialised.

Computation of CDF⁻¹
~~~~~~~~~~~~~~~~~~~~

In general we have some integrant, for which we need to find the inverse
cumulative function. Given a DCS, we define an integrant `DCS_int`, and solve
like this:

    cumulative_dcs(a) := integrate(lower_limit, a, DCS_int)
    tcs = cumulative_dcs(upper_limit)
    spec_tcs(log(K)) := log(tcs)
    icdf(K, P) -> a | P == cumulative_dcs(a)/tcs

We may do the inversion using Newton-Raphson method.

`set_elastic_data(K, dcs(theta): f -> f)`
-----------------------------------------
    dcs_int(theta) := dcs(theta) 2 pi sin(theta)
    cumulative_dcs(a) := integrate(0, a, dcs_int)
    tcs = cumulative_dcs(pi) # compute total for normalisation
    elastic_tcs(log(K)) := log(tcs)
    icdf(K, P) -> theta | P == cumulative_dcs(theta)/tcs

`set_inelastic_data(K, dcs(w_0): f -> f)`
-----------------------------------------
    # concerning integration range: 0 < w_0 < K
    cumulative_dcs(a) := integrate(0, a, dcs)
    tcs = cumulative_dcs(K)
    inelastic_tcs(log(K)) := log(tcs)
    icdf(K, P) -> w_0 | P == cumulative_dcs(w_0)/tcs

`set_ionization_data(B, tcs(K): f -> f)`
----------------------------------------
    loglog_tcs_map(log(K)) := log(tcs(K))
    ionization_tcs(B) := loglog_tcs_map

The material class has some more helper functions:
* ionization_energy(K, P) -> creates an ionization_map interpolating the
    tables in ionization_tcs using the log of K, if K > B. Each cross-section
    found is added to a running total tcs: ionization_map(tcs) = B. Then the
    first binding energy for which P < tcs(B)/TCS. This is a weird way of choosing
    a binding energy with a probability that scales with the associated cross-
    section.

* outer_shell_ionization_energy(w_0) ->
    for each binding energy B
        if B < 100 eV and w_0 > B
            return B
    return -1
    
Why the upper limit of 100 eV?


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
