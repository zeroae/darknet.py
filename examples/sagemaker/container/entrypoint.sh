#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
conda_environment=$(python -c '
from pynvml import *
try:
  nvmlInit()
  if nvmlDeviceGetCount() > 0:
    print("gpu")
except:
  print("cpu")
')
conda activate $conda_environment
exec "$@"
