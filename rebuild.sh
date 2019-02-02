#!/bin/bash


python setup.py bdist_wheel --universal
pip install --upgrade --ignore-installed dist/magenta*.whl
