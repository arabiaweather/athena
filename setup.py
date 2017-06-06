from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='athena',
    version='0.0.1',
    description='Automatic equation building using machine-learning based curve-fitting techniques.',
    long_description=long_description,
    url='https://github.com/arabiaweather/athena',
    author='Khaled Sharif',
    author_email='khaled.sharif@arabiaweather.com',
    keywords='machine-learning curve-fitting data-science',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'notebooks', 'examples']),
    install_requires=['pandas', 'tensorflow', 'sklearn', 'sympy'],
)
