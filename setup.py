from setuptools import setup

setup(
    name="seec",
    version="1.0.1",
    description="Implementation of SEEC-nt in python, adapted from git@github.com:morcoslab/SEEC-nt.git",
    author="Jonathan Martin, Alberto de la Paz",
    author_email="jonathan.martin3@utdallas.edu",
    license="MIT",
    packages=["seec"],
    zip_safe=False,
    install_requires=[
        "biopython",
        "matplotlib",
        "numpy",
        "numba",
        "scipy",
        "dca @ git+https://github.com/utdal/py-mfdca@v1.0.0",
    ],
)
