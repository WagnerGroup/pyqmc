'''
Real space quantum Monte Carlo calculations using pyscf
'''
import setuptools
from distutils.core import setup
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def setup_pyqmc():
    setup(
        name='pyqmc',
        maintainer='Lucas Wagner',
        maintainer_email='lkwagner@illinois.edu',
        packages=['pyqmc'],
        version=get_version("pyqmc/__init__.py"),
        package_data={'pyqmc': ['data/*.pkl']},
        license='MIT License',
        url='https://github.com/WagnerGroup/pyqmc',
        description='Python library for real space quantum Monte Carlo',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        install_requires=[
            "scipy",
            "pandas",
            "pyscf",
            "h5py"
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'License :: OSI Approved :: MIT License',
        ],
    )


if __name__ == '__main__':
    setup_pyqmc()
