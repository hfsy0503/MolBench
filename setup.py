# !/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='molbench',
    version='0.1.0',
    description='MolBench - 分子机器学习基准测试框架',
    author='MolBench Contributors',
    author_email='2310753@mail.nankai.edu.cn',
    url='https://github.com/hfsy0503/molbench',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'pyyaml>=5.3.0',
        'rdkit>=2021.03.1',
        'torch>=1.9.0',
        'deepchem>=2.7.0',     
    ],
    entry_points={
        'console_scripts': [
            'molbench=molbench.cli:main',
        ],
    },
    keywords=[
        'molecular-machine-learning',
        'benchmark',
        'cheminformatics',
        'deep-learning',
        'sklearn',
        'graph-neural-networks',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
