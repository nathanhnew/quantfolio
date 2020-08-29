from setuptools import setup, find_packages

setup(
    name="quantfolio",
    version="0.1.0",
    author="Nathan New",
    author_email="nathanhnew@gmail.com",
    description="Python Portfolio Optimization Tool",
    long_description="Tool for optimizing stock portfolios based on variety of metrics",
    url="https://github.com/nathanhnew/quantfolio",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requirements=">=3.6"
)