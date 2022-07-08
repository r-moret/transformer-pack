from setuptools import find_packages, setup

setup(
    name="transformer-pack",
    packages=find_packages(include=["transformer-pack"]),
    version="0.1.0",
    description="Package of extra scikit-learn transformers",
    author="Rafael Moret",
    license="GPLv3",
    install_requires=["numpy", "pandas", "scikit-learn", "tensorflow"],
)
