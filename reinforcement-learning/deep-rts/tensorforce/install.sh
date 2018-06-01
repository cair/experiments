#!/usr/bin/env bash

pip3 install virtualenv
mkdir venv
python3 -m virtualenv venv
source ./venv/bin/activate

pip3 install tensorflow
pip3 install tensorflow-gpu
pip3 install git+https://github.com/reinforceio/tensorforce.git