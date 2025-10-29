#!/bin/bash

LMOD_DISABLE_SAME_NAME_AUTOSWAP=no module load Python/3.12.3-GCCcore-13.3.0
LMOD_DISABLE_SAME_NAME_AUTOSWAP=no module load uv/0.2.30-GCCcore-13.3.0
LMOD_DISABLE_SAME_NAME_AUTOSWAP=no module load cuDNN/9.5.0.50-CUDA-12.6.0
source .venv/bin/activate
#bash -i $@