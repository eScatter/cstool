#!/usr/bin/env python

from setuptools import setup, Extension
from os import path
from codecs import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cstool',
    version='0.1.0',
    long_description=long_description,
    description='Computes cross-sections from several sources and'
                ' compiles them into a material file for the e-Scatter'
                ' simulator.',
    license='Apache v2',
    author='Johan Hidding',
    author_email='j.hidding@esciencecenter.nl',
    url='https://github.com/eScatter/pyelsepa.git',
    packages=['cstool'],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'],
    install_requires=[
        'pint', 'numpy', 'cslib', 'pyelsepa', 'noodles', 'tinydb',
        'ruamel.yaml'],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pep8', 'pyflakes', 'sphinx'],
    },
    # https://docs.python.org/3.6/distutils/setupscript.html#describing-extension-modules
    # http://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
    # http://stackoverflow.com/questions/16854066/using-distutils-and-build-clib-to-build-c-library
    ext_modules=[Extension('icdf', ['src/icdf.cc'])]
    # libraries=[('icdf', {'sources': ['src/icdf.cc']})],
)
