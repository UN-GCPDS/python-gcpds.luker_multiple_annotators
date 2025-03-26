from setuptools import setup, find_packages

setup(
    name="ma_models",  # Mandatory
    version="0.1.0",  # Recommended
    packages=find_packages(where='src'),  # Recommended
    install_requires=[],  # Recommended, even if empty
    package_dir={'':'src'}
)