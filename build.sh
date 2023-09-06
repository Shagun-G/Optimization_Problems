#!/bin/bash

pip install wheel
python setup.py bdist_wheel sdist
pip install .