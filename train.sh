#!/usr/bin/env bash 

# Necessary exports
export PYTHONPATH=${PWD}/src:$PYTHONPATH

# Training
python src/train.py
