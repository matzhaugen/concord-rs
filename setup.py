from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='pyconcord',
    version='0.1.0',
    description='Example of python-extension using rust-numpy',
    rust_extensions=[RustExtension(
        'pyconcord.pyconcord',
        './Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
    zip_safe=False,
)