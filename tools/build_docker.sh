#!/usr/bin/env bash

set -uex
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$WORK_DIR"

docker_repo="zhengqinwen/foot_status_prediction:v1.0"
docker build -t ${docker_repo} -f Dockerfile .
# docker push ${docker_repo}
