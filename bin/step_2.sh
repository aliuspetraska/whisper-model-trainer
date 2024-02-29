#!/bin/bash

echo "[1] DEEPSPEED"

pip3 install deepspeed -U

ds_report

echo "[2] NVIDIA-CUDA-TOOLKIT"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt install cuda-toolkit

ds_report