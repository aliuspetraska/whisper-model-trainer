#!/bin/sh

export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DS_BUILD_CPU_ADAM=1
export BUILD_UTILS=1