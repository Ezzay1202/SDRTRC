#!/usr/bin/env bash
set -e

for pred_len in 96 192 336 720; do
  echo "============================================================"
  echo "Check validation alignment pred_len=${pred_len}"
  echo "============================================================"
  python tools/check_val_alignment.py \
    --base_dir results/electricity_96_${pred_len}_XLinear_custom_ftM_sl96_ll48_pl${pred_len}_dm2048_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
    --ours_dir results/Electricity_SDRTR_96_${pred_len}_SDRTR_custom_ftM_sl96_ll48_pl${pred_len}_dm2048_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_SDR_TR_0 \
    --prefix val

done
