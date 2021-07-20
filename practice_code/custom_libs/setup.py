from setuptools import setup

requirements = ["numpy", "scikit-learn==0.24.2"]

setup(
    name='custom_data_transformers',
    version='0.1',
    description='library for custom data processing',
    license="Proprietary",
    classifiers=['License :: Other/Proprietary License'],
    packages=['custom_data_transformers'],
    install_requires=requirements,
    include_package_data=True,
)