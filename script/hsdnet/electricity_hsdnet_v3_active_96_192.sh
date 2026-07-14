#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity_HSDNet_v3_active
model_name=HSDNet
data_name=custom

mkdir -p logs/LongForecasting/Electricity

for pred_len in 96 192; do
  if [ "$pred_len" = "96" ]; then
    d_model=2048
    t_ff=512
    c_ff=32
    c_dropout=0.1
    t_dropout=0.2
    head_dropout=0.2
    embed_dropout=0.2
    learning_rate=0.0002
    batch_size=16
  elif [ "$pred_len" = "192" ]; then
    d_model=2048
    t_ff=512
    c_ff=24
    c_dropout=0.1
    t_dropout=0.2
    head_dropout=0.3
    embed_dropout=0.1
    learning_rate=0.0001
    batch_size=16
  fi

  python -u run_longExp.py \
    --random_seed 2025 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len} \
    --model $model_name \
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
    --hsd_state_dim 32 \
    --hsd_dep_dim 16 \
    --hsd_num_groups 4 \
    --hsd_topk 16 \
    --hsd_zero_init 1 \
    --hsd_rng_safe_init 1 \
    --hsd_disable_dep 0 \
    --hsd_fusion residual \
    --hsd_gate_init -4.0 \
    --hsd_dropout 0.05 \
    --hsd_residual_scale 0.08 \
    --hsd_aux_lambda 0.001 \
    --hsd_residual_target_lambda 0.05 \
    --hsd_residual_target_clip 3.0 \
    --hsd_residual_target_risk_weight 1 \
    --hsd_residual_target_min_weight 0.10 \
    --hsd_risk_use_quantile 1 \
    --hsd_risk_quantile 0.70 \
    --hsd_risk_sharpness 8.0 \
    --hsd_risk_floor 0.0 \
    --hsd_detach_risk 1 \
    --hsd_debug 1 \
    --des 'HSDNet_v3_active' \
    --train_epochs 30 \
    --patience 3 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --batch_size $batch_size \
    --learning_rate $learning_rate | tee logs/LongForecasting/Electricity/${model_name}_${model_id_name}_${pred_len}_active.log

done
