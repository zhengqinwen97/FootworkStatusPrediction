#!/usr/bin/env bash

set -uex
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run -it --rm \
  --gpus all \
  -p 6006:6006 \
  -v /home/qizheng/workspace/FootworkStatusPrediction:/workspace \
  zhengqinwen/foot_status_prediction:v1.0 \
  /bin/bash