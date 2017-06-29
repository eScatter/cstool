#!/usr/bin/env python

from distutils.core import setup
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
    package_data={'cstool': ['data/endf_sources.json']},
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'],
    install_requires=[
        'pint==0.8.1', 'numpy==1.12.1', 'cslib==0.1.0', 'pyelsepa==0.1.1', 'noodles==0.2.3', 'tinydb==3.3.1',
        'ruamel.yaml==0.15.15'],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pep8', 'pyflakes'],
        'dev': ['pytest', 'pytest-cov', 'pep8', 'pyflakes', 'sphinx'],
    }
)
