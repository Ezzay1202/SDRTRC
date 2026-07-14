#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

root_path_name=${root_path_name:-./DataSets/hogprice/}
data_path_name=${data_path_name:-hogprice260515.csv}
model_id_name=HogPrice_HSDNet_v2
model_name=HSDNet
data_name=custom

mkdir -p logs/LongForecasting/hogprice

for pred_len in 30 90 180; do
  python -u run_longExp.py \
    --random_seed 2025 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_96_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features MS \
    --target hogprice \
    --freq d \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 24 \
    --dec_in 24 \
    --c_out 1 \
    --d_model 32 \
    --t_ff 128 \
    --c_ff 24 \
    --t_dropout 0.05 \
    --c_dropout 0.05 \
    --embed_dropout 0.05 \
    --head_dropout 0.1 \
    --hsd_state_dim 32 \
    --hsd_dep_dim 16 \
    --hsd_num_groups 4 \
    --hsd_gate_init -4.0 \
    --hsd_dropout 0.05 \
    --hsd_topk 8 \
    --hsd_zero_init 1 \
    --hsd_rng_safe_init 1 \
    --hsd_residual_scale 0.08 \
    --hsd_aux_lambda 0.003 \
    --hsd_risk_use_quantile 1 \
    --hsd_risk_quantile 0.65 \
    --hsd_risk_sharpness 8.0 \
    --hsd_debug 1 \
    --des 'HSDNet_v2' \
    --train_epochs 50 \
    --patience 10 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0005 | tee logs/LongForecasting/hogprice/${model_name}_${model_id_name}_${pred_len}_v2.log

done
