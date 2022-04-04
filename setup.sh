#!/usr/bin/env bash

conda create -n goemotion python=3.7
conda activate goemotion

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers
pip install attrdict==2.0.1