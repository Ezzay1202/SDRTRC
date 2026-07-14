#!/usr/bin/env bash
set -e

mkdir -p analysis_outputs

for pred_len in 336 720; do
  python tools/calibrate_residual_beta.py \
    --base_result_dir results/electricity_96_${pred_len}_XLinear_custom_ftM_sl96_ll48_pl${pred_len}_dm2048_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
    --ours_result_dir results/Electricity_SDRTR_96_${pred_len}_SDRTR_custom_ftM_sl96_ll48_pl${pred_len}_dm2048_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_SDR_TR_0 \
    --mode scalar \
    --clip_min 0.0 \
    --clip_max 1.0 \
    --out_dir analysis_outputs/SDRTRC_Electricity_${pred_len}_scalar

done
