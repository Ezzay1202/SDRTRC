#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

root_path_name=${root_path_name:-./dataset/electricity/}
data_path_name=${data_path_name:-electricity.csv}
model_id_name=Electricity_HSDNet_v2
model_name=HSDNet
data_name=custom

mkdir -p logs/LongForecasting/electricity

for pred_len in 96 192; do
  python -u run_longExp.py \
    --random_seed 2025 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 16 \
    --t_ff 1 \
    --c_ff 1 \
    --t_dropout 0 \
    --c_dropout 0 \
    --embed_dropout 0.1 \
    --head_dropout 0 \
    --hsd_state_dim 32 \
    --hsd_dep_dim 16 \
    --hsd_num_groups 4 \
    --hsd_gate_init -5.0 \
    --hsd_dropout 0.05 \
    --hsd_topk 16 \
    --hsd_zero_init 1 \
    --hsd_rng_safe_init 1 \
    --hsd_residual_scale 0.05 \
    --hsd_aux_lambda 0.005 \
    --hsd_risk_use_quantile 1 \
    --hsd_risk_quantile 0.70 \
    --hsd_risk_sharpness 8.0 \
    --hsd_debug 1 \
    --des 'HSDNet_v2' \
    --train_epochs 30 \
    --patience 10 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0005 | tee logs/LongForecasting/electricity/${model_name}_${model_id_name}_${pred_len}_v2.log

done
