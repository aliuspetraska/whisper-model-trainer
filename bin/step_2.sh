#!/bin/bash

echo "[1] DEEPSPEED"

pip3 install deepspeed -U

ds_report

echo "[2] NVIDIA-CUDA-TOOLKIT"

sudo apt-get install nvidia-cuda-toolkit

