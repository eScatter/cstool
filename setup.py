#!/usr/bin/env python

from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    has_cython = False
else:
    has_cython = True

from os import path
from codecs import open


if has_cython:
    ext_modules = cythonize([Extension(
            "cstool.icdf",
            sources=["src/icdf.cc", "cstool/icdf.pyx"],
            language="c++",
            extra_compile_args=["-std=c++11"])])
else:
    ext_modules = [Extension(
            "cstool.icdf",
            sources=["src/icdf.cc", "cstool/icdf.cpp"],
            language="c++",
            extra_compile_args=["-std=c++11"])]

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
        'pint==0.7.2', 'numpy', 'cslib', 'pyelsepa', 'noodles', 'tinydb',
        'ruamel.yaml'],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pep8', 'pyflakes'],
        'dev': [
            'pytest', 'pytest-cov', 'pep8', 'pyflakes', 'sphinx',
            'cython'],
    },
    ext_modules=ext_modules
)
