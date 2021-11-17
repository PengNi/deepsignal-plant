from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re

here = os.path.abspath(os.path.dirname(__file__))


# Get the version number from _version.py, and exe_path (learn from tombo)
verstrline = open(os.path.join(here, 'deepsignal_plant', '_version.py'), 'r').readlines()[-1]
vsre = r"^DEEPSIGNAL_PLANT_VERSION = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "deepsignal_plant/_version.py".')


# def find_version(*file_paths):
#     version_file = read(*file_paths)
#     version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
#                               version_file, re.M)
#     if version_match:
#         return version_match.group(1)
#     raise RuntimeError("Unable to find version string.")


# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
#         self.test_suite = True
#
#     def run_tests(self):
#         import pytest
#         errno = pytest.main(self.test_args)
#         sys.exit(errno)


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


long_description = read('README.rst')


with open('requirements.txt', 'r') as rf:
    required = rf.read().splitlines()


setup(
    name='deepsignal-plant',
    packages=['deepsignal_plant'],
    keywords=['methylation', 'nanopore', 'neural network'],
    version=__version__,
    url='https://github.com/PengNi/deepsignal-plant',
    download_url='https://github.com/PengNi/deepsignal-plant/archive/{}.tar.gz'.format(__version__),
    license='GNU General Public License v3 (GPLv3)',
    author='Peng Ni',
    # tests_require=['pytest'],
    # which package needs 'future' package?
    # install_requires=['numpy>=1.15.3',
    #                   'h5py>=2.8.0',
    #                   'statsmodels>=0.9.0',
    #                   'scikit-learn>=0.20.1',
    #                   'torch>=1.2.0,<=1.7.0',
    #                   ],
    install_requires=required,
    # cmdclass={'test': PyTest},
    author_email='543943952@qq.com',
    description='A deep-learning method for detecting DNA methylation state '
                'from Oxford Nanopore sequencing reads of plants',
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'deepsignal_plant=deepsignal_plant.deepsignal_plant:main',
            ],
        },
    platforms='any',
    # test_suite='test',
    zip_safe=False,
    include_package_data=True,
    # package_data={'deepsignal': ['utils/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        ],
    # extras_require={
    #     'testing': ['pytest'],
    #   },
)
