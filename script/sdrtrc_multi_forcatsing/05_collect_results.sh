#!/usr/bin/env bash
set -e
python tools/sdrtrc_multi_runner.py \
  --stage collect \
  --beta_mode ${BETA_MODE:-scalar}
