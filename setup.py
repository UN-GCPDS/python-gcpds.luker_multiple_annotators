from setuptools import setup, find_packages

# setup(
#     name="ma_models",  # Mandatory
#     version="0.1.0",  # Recommended
#     packages=find_packages(where='src'),  # Recommended
#     install_requires=['gpflow'],  # Recommended, even if empty
#     package_dir={'':'src'}
# )

setup(
    name="gcpds_luker_multiple_annotators", 
    version="0.1.0",
    packages=find_packages(where='src'), 
    install_requires=[
        'torch',
        'gpflow',
        'optuna',
        'numpy',
        'scikit-learn',
        'pytorch-tabnet',
    ],
    package_dir={"": "src"},
)
