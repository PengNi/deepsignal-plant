from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


long_description = read('README.rst')


setup(
    name='deepsignal2',
    packages=['deepsignal2'],
    keywords=['methylation', 'nanopore', 'neural network'],
    version=find_version('deepsignal', '__init__.py'),
    url='',
    download_url='',
    license='GNU General Public License v3 (GPLv3)',
    author='Peng Ni, Neng Huang',
    # tests_require=['pytest'],
    install_requires=['numpy>=1.15.3',
                      'h5py>=2.8.0',
                      'statsmodels>=0.9.0',
                      'scikit-learn>=0.20.1',
                      ],
    # cmdclass={'test': PyTest},
    author_email='543943952@qq.com',
    description='a',
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'deepsignal2=deepsignal2.deepsignal2:main',
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
    # scripts=['deepsignal/deepsignal.py'],
)
