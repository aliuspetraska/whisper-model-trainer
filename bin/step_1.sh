#!/bin/bash

echo "[1] UPDATING"

sudo apt-get update -y
sudo apt-get upgrade -y

echo "[2] ADDING/ENABLING ADDITIONAL REPOS"

sudo apt-get install gcc clang cmake procps nano python3-pip python3-dev git git-lfs ffmpeg libaio-dev

echo "[3] DEV PREREQUISITES"

pip3 install setuptools pip wheel -U
git lfs install

echo "[4] NVIDIA"

# https://www.cherryservers.com/blog/install-cuda-ubuntu

sudo apt-get install ubuntu-drivers-common
sudo apt-get install nvidia-driver-545

sudo reboot