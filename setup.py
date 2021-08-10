"""
Copyleft 2021
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz-Jaramillo
"""

from setuptools import setup, find_packages


setup(
    name='iFAS',  # Required
    version='2021-v1',  # Required
    description='image Fidelity Assessment Software',  # Optional
    url='https://github.com/bortizj/iFAS',  # Optional
    author='Benhur Ortiz-Jaramillo',  # Optional
    author_email='benhur.ortizjaramillo@gmail.com',  # Optional
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(exclude=['test', 'docs', 'example_data']),  # Required
)