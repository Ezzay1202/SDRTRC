#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT_PATH=${ROOT_PATH:-/Users/ezzay/PycharmProjects/MUSE-main/dataset}
python tools/sdrtrc_multi_runner.py \
  --stage export_xlinear \
  --datasets ${DATASETS:-all} \
  --horizons ${HORIZONS:-all} \
  --root_path $ROOT_PATH \
  --gpu ${CUDA_VISIBLE_DEVICES%%,*} \
  --num_workers ${NUM_WORKERS:-4} \
  --eval_drop_last ${EVAL_DROP_LAST:-1} \
  --eval_split both
python tools/sdrtrc_multi_runner.py \
  --stage export_sdrtr \
  --datasets ${DATASETS:-all} \
  --horizons ${HORIZONS:-all} \
  --root_path $ROOT_PATH \
  --gpu ${CUDA_VISIBLE_DEVICES%%,*} \
  --num_workers ${NUM_WORKERS:-4} \
  --eval_drop_last ${EVAL_DROP_LAST:-1} \
  --eval_split both
