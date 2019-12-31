import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'requirements.txt'), 'rt') as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name='diffrascape',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='BSD',
    author='jlynch',
    author_email='jlynch@bnl.gov',
    description='',
    install_requires=requirements
)
