CSTool
======

This tool compiles cross-sections for low-energy electron scattering in a material
into a `.mat` file that is understood by the e⁻Scatter simulator.

If you need to change anything in the physics models used by the simulator, this is
the place to do it.

Currently, this module needs both the `pyelsepa` and `cslib` packages to be installed,
preferably in a VirtualEnv (so that you don't muck-up your Python installation).

.. image:: https://travis-ci.org/eScatter/cstool.svg?branch=master
    :target: https://travis-ci.org/eScatter/cstool

Installing
==========

To setup an environment for CSTool, first make sure you have the correct version of Python (these commands may vary slightly from system to system)::

    > python3 --version   # version should be 3.5 or higher
    Python 3.5.3

clone the needed repositories::

    > git clone https://github.com/eScatter/cslib.git
    ...
    > git clone https://github.com/eScatter/pyelsepa.git
    ...
    > git clone https://github.com/eScatter/cstool.git
    ...

create a virtual environment::

    > python3 -m venv cstool-env
    ...
    > source ./cstool-env/bin/activate
    (cstool-env) >

You may need to update `pip` at this point::

    > pip install --upgrade pip

Now we need to install `cslib` and `pyelsepa`::

    (cstool-env) > cd cslib
    (cstool-env) ./cslib> pip install .
    (cstool-env) ./cslib> cd ../pyelsepa
    (cstool-env) ./pyelsepa> pip install .
    (cstool-env) ./pyelsepa>

And we have to install the Docker image for Elsepa, make sure you have downloaded `adus_v1_0.tar.gz`_::

    (cstool-env) ./pyelsepa> docker -v  # make sure you have the latest
    Docker version 1.13.1, build 092cba372
    (cstool-env) ./pyelsepa> cd docker
    (cstool-env) ./pyelsepa> ls
    adus_v1_0.tar.gz  Dockerfile  README.md
    (cstool-env) ./pyelsepa> docker build -t elsepa .
    ...
    Successfully built <some hex-code>


Now we need to install `cstool`::

    (cstool-env) ./pyelsepa> cd ../cstool
    (cstool-env) ./cstool> pip install .
    
Now you should be ready to run the example in `examples/cs.py`::

    (cstool-env) ./cstool> python examples/cs.py data/materials/silicon.yaml

Data sources
============

The material files contain most of the physics involved.

* [ELSEPA](http://adsabs.harvard.edu/abs/2005CoPhC.165..157S) can be downloaded from the
  Computer Physics communications Program library as `adus_v1_0.tar.gz`_. It has an
  attribute-only license for non-commercial use. We use a dockerized ELSEPA to compute Mott
  cross-sections, through the `pyElsepa`_ module.

* Livermore database [ENDF/B-VII.1](http://www.nndc.bnl.gov/endf/b7.1/download.html). We use this
  database retrieve ionization energies, occupancy and cross-sections.

* ELF data (Energy Loss Function). This data was compiled by Kieft & Bosch (2008) for their
  (Geant4 based) version of the electron scattering model. It uses data from Palik (1985,1998) -
  "Handbook of Optical Constants of Solids" [1]_ - and [Henke et al.](henke.lbl.gov) [2]_. This data
  is included here in the `/data/elf` folder.

* Phonon scattering is computed by recipe from Schreiber & Fitting [3]_.

References
==========

.. [1] Palik, Edward D. Handbook of optical constants of solids. Vol. 3. Academic press, 1998.

.. [2] B.L. Henke, E.M. Gullikson, and J.C. Davis. X-ray interactions: photoabsorption, scattering, transmission, and reflection at E=50-30000 eV, Z=1-92, Atomic Data and Nuclear Data Tables Vol. 54 (no.2), 181-342 (July 1993).

.. [3] Schreiber, E., and H-J. Fitting. "Monte Carlo simulation of secondary electron emission from the insulator SiO 2." Journal of Electron Spectroscopy and Related Phenomena 124.1 (2002): 25-37.

.. _`adus_v1_0.tar.gz`: http://www.cpc.cs.qub.ac.uk/summaries/ADUS_v1_0.html
.. _`pyElsepa`: http://github.com/eScatter/pyelsepa.git
