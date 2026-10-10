#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated wrapper] forwarding to the frozen root protocol:" >&2
echo "python train.py --model UAFL --dataset PaviaU --mode mixed" >&2
python train.py --model UAFL --dataset PaviaU --mode mixed "$@"
