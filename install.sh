#!/usr/bin/env bash

ENV_NAME=3DOR_2024

conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Install PyTorch, check your system CUDA version etc. 
# Find wheels at https://download.pytorch.org/whl/torch_stable.html
# pip install https://download.pytorch.org/whl/cu117/torch-2.0.1%2Bcu117-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt
pip install -e .
ipython kernel install --name $ENV_NAME --user