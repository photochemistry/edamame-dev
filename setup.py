#!/usr/bin/env python

# from distutils.core import setup, Extension
import os

from setuptools import find_packages, setup

#Copied from wheel package
here = os.path.abspath(os.path.dirname(__file__))

long_desc = "".join(open("README.md").readlines())

setup(
    name='edamame-dev', # the package name
    version="0.3.1",
    description='Edamame-kun.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
    author='Masakazu Matsumoto',
    author_email='vitroid@gmail.com',
    url='https://github.com/photochemistry/edamame/',
    keywords=['edamame', 'photochemistry'],

    packages=find_packages(),
    # package_dir = {'': 'dev'},
    # entry_points = {
    #     'genice2_format': [
    #         'svg = genice2_svg.formats.svg',
    #         'png = genice2_svg.formats.png',
    #     ],
    # },
    install_requires=['numpy', 'geopandas', 'pandas', 'scipy', 'shapely'],

    license='MIT',
)

# Colabからデータを見えるようにするには、インターネット上に置くしかなさそうだ。
# 暫定的に化学科サーバに置くか。
