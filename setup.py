# -*- coding: utf-8 -*-
"""
Setup script for QDebrisFlow.

Note: QDebrisFlow is a QGIS plugin and is normally installed by copying the
plugin folder into the QGIS plugins directory (see README). This setup.py is
provided for packaging/metadata purposes and to declare the pip-installable
dependencies (numpy, numba). PyQt5, the QGIS Python API and GDAL are provided
by the QGIS installation and are not installed here.
"""

from setuptools import setup, find_packages

setup(
    name="QDebrisFlow",
    version="1.0",
    description=(
        "Modelling of post-fire debris flows (O'Brien quadratic rheology + LIA) "
        "as a QGIS plugin."
    ),
    author="Aidar Khaybulin",
    license="MIT",
    url="https://github.com/your/repo",
    python_requires=">=3.7",
    packages=find_packages(
        include=[
            "QDebrisFlowEN", "QDebrisFlowEN.*",
            "QDebrisFlowRU", "QDebrisFlowRU.*",
        ]
    ),
    package_data={
        "QDebrisFlowEN": ["metadata.txt", "icon.png"],
        "QDebrisFlowRU": ["metadata.txt", "icon.png"],
    },
    install_requires=[
        "numpy",
        "numba",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
