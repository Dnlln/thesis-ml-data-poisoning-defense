"""Установочный скрипт пакета poisondefense."""

from setuptools import find_packages, setup

setup(
    name="poisondefense",
    version="1.0.0",
    description=(
        "Программный модуль защиты классических моделей машинного "
        "обучения от атак отравления данных"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Dnlln",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
