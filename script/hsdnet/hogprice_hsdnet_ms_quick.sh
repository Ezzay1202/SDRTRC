#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=HSDNet
root_path_name=./DataSets/hogprice/
data_path_name=hogprice260515.csv
model_id_name=HogPrice_HSDNet_MS_QUICK
data_name=custom

mkdir -p logs/LongForecasting/hogprice

for pred_len in 30 90 180; do
  python -u run_longExp.py \
    --random_seed 2021 \
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
    --d_model 16 \
    --t_ff 1 \
    --c_ff 1 \
    --embed_dropout 0.1 \
    --head_dropout 0 \
    --hsd_state_dim 32 \
    --hsd_dep_dim 16 \
    --hsd_num_groups 4 \
    --hsd_gate_init -3.0 \
    --hsd_dropout 0.1 \
    --hsd_topk 0 \
    --hsd_fusion interp \
    --des 'HSDNet_v1' \
    --train_epochs 20 \
    --patience 5 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0005 | tee logs/LongForecasting/hogprice/${model_name}_${model_id_name}_${pred_len}.log
done
