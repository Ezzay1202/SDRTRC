#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

root_path_name=${root_path_name:-./dataset/}
data_path_name=${data_path_name:-ETTh1.csv}
model_id_name=ETTh1_HSDNet_v11
model_name=HSDNet
data_name=ETTh1

mkdir -p logs/LongForecasting/ETTh1

for pred_len in 96 720; do
  if [ "$pred_len" = "96" ]; then
    d_model=128
    t_ff=512
    c_ff=7
    t_dropout=0
    c_dropout=0
    head_dropout=0.2
    embed_dropout=0
  else
    d_model=64
    t_ff=128
    c_ff=7
    t_dropout=0.1
    c_dropout=0
    head_dropout=0.2
    embed_dropout=0
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
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
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
    --hsd_gate_init -3.0 \
    --hsd_dropout 0.0 \
    --hsd_topk 0 \
    --hsd_temperature 1.0 \
    --hsd_fusion residual \
    --hsd_zero_init 1 \
    --hsd_rng_safe_init 1 \
    --hsd_disable_dep 0 \
    --des 'HSDNet_v11' \
    --train_epochs 30 \
    --patience 10 \
    --gpu 0 \
    --num_workers 4 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 | tee logs/LongForecasting/ETTh1/${model_name}_${model_id_name}_${pred_len}.log
done
