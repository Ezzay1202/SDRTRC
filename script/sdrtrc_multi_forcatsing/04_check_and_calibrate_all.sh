#!/usr/bin/env bash
set -e
python tools/sdrtrc_multi_runner.py \
  --stage check \
  --datasets ${DATASETS:-all} \
  --horizons ${HORIZONS:-all} \
  --beta_mode ${BETA_MODE:-scalar}
python tools/sdrtrc_multi_runner.py \
  --stage calibrate \
  --datasets ${DATASETS:-all} \
  --horizons ${HORIZONS:-all} \
  --beta_mode ${BETA_MODE:-scalar} \
  --check_alignment 1
