#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cpu
exec "$@"
