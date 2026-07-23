#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd /workspace/SDRTRC-main
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="log/v32_validation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "=== SDR-TR v3.1 Validation | GPU=$GPU_ID | $(date) ==="
for DS in ETTh1 ETTh2; do
  for PL in 96 192; do
    LOG="$LOG_DIR/v31_${DS}_${PL}.log"
    echo ">>> $DS pred_len=$PL  $(date)"
    python -u run_longExp.py \
      --is_training 1 \
      --model SDRTR_v32 \
      --model_id "${DS}_${PL}" \
      --data $DS \
      --root_path ./dataset/ \
      --data_path "${DS}.csv" \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PL \
      --enc_in 7 \
      --d_model 256 \
      --t_ff 512 \
      --c_ff 512 \
      --train_epochs 10 \
      --patience 3 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --lradj type1 \
      --des 'v32_validation' \
      --itr 1 \
      --usenorm 1 \
      --use_gpu 1 \
      --gpu 0 \
      --sdr_debug 1 \
      --sdr_state_dim 128 \
      > "$LOG" 2>&1
    echo "    Done: $DS pred_len=$PL  $(date)"
  done
done
echo "=== ALL DONE: $(date) ==="
echo "Logs: $LOG_DIR"
