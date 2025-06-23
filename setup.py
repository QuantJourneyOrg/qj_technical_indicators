"""
QuantJourney Technical-Indicators - Package Setup
=================================================
Legacy setuptools build script for environments that do not yet leverage
`pyproject.toml`. Mirrors metadata defined in the modern build backend so that
classic workflows (e.g. `pip install .` or `python setup.py sdist bdist_wheel`)
continue to work.

Author: Jakub Polec  <jakub@quantjourney.pro>
License: MIT
"""
from setuptools import setup, find_packages

setup(
    name="quantjourney-ti",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=2.0.0",
        "numba>=0.59.0",
        "matplotlib>=3.5.0"
    ],
    author="Jakub Polec",
    author_email="jakub@quantjourney.pro",
    description="High-performance technical indicators library",
    license="MIT"
)