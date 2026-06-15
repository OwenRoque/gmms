#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="gmss",
    version="0.0.1",
    description="Guided Music Source Separation",
    install_requires=[
        "lightning",
        "hydra-core",
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "pyrootutils",
    ],
    packages=find_packages(where="."),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "test_separation = src.test_separation:main",
        ]
    },
)
