from setuptools import setup, find_packages

setup(
    name='Opt_Problems',
    version='0.0.1',
    description='Sample Optimization Problems',
    # package_dir={"":"Opt_Problems"},
    packages=find_packages(exclude=["test*"]),
    author='Shagun Gupta',
    url='https://github.com/Shagun-G/Optimization_Problems',
    license="MIT",
    install_requires=[
        'numpy',
    ],
    # python_requires=">3.10",
)