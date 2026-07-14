#!/usr/bin/env bash
set -e

# Re-export aligned val/test predictions from existing checkpoints.
# It does NOT retrain. It only loads checkpoints and writes val_pred/val_true/pred/true.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root_path_name=/home/data/zsh/seq_data
data_path_name=electricity.csv
data_name=custom

for pred_len in 336 720; do
  d_model=2048; t_ff=512; c_ff=24
  c_dropout=0.1; t_dropout=0.2; head_dropout=0.3; embed_dropout=0.1
  learning_rate=0.0001; batch_size=16

  echo "============================================================"
  echo "Export XLinear val/test pred_len=${pred_len}"
  echo "============================================================"
  python -u run_longExp.py \
    --random_seed 2025 \
    --is_training 0 \
    --eval_split both \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id electricity_96_${pred_len} \
    --model XLinear \
    --data $data_name \
    --features M \
    --target OT \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model $d_model \
    --t_ff $t_ff \
    --c_ff $c_ff \
    --t_dropout $t_dropout \
    --c_dropout $c_dropout \
    --embed_dropout $embed_dropout \
    --head_dropout $head_dropout \
    --des 'Exp' \
    --train_epochs 30 \
    --patience 3 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --eval_drop_last 1 \
    --batch_size $batch_size \
    --learning_rate $learning_rate

  echo "============================================================"
  echo "Export SDR-TR val/test pred_len=${pred_len}"
  echo "============================================================"
  python -u run_longExp.py \
    --random_seed 2025 \
    --is_training 0 \
    --eval_split both \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id Electricity_SDRTR_96_${pred_len} \
    --model SDRTR \
    --data $data_name \
    --features M \
    --target OT \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model $d_model \
    --t_ff $t_ff \
    --c_ff $c_ff \
    --t_dropout $t_dropout \
    --c_dropout $c_dropout \
    --embed_dropout $embed_dropout \
    --head_dropout $head_dropout \
    --sdr_state_dim 32 \
    --sdr_dep_dim 16 \
    --sdr_num_groups 4 \
    --sdr_topk 16 \
    --sdr_zero_init 1 \
    --sdr_rng_safe_init 1 \
    --sdr_disable_dep 0 \
    --sdr_dropout 0.05 \
    --sdr_residual_scale 0.08 \
    --sdr_aux_lambda 0.001 \
    --sdr_trust_logit -4.0 \
    --sdr_use_learnable_gate 0 \
    --sdr_residual_target_lambda 0.05 \
    --sdr_risk_use_quantile 1 \
    --sdr_risk_quantile 0.70 \
    --sdr_risk_sharpness 8.0 \
    --sdr_risk_floor 0.0 \
    --sdr_detach_risk 1 \
    --sdr_debug 1 \
    --des 'SDR_TR' \
    --train_epochs 30 \
    --patience 3 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --eval_drop_last 1 \
    --batch_size $batch_size \
    --learning_rate $learning_rate

done
