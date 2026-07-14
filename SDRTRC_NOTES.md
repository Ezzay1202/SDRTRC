# SDR-TRC: State-Dependent Residual Learning with Trust-Region Calibration

This package contains the SDR-TR / SDR-TRC implementation based on XLinear.

## Method

Base prediction:

```text
Y_base = XLinear(X)
```

SDR-TR prediction:

```text
Y_sdrtr = Y_base + Delta_TR
```

SDR-TRC calibrated prediction:

```text
Y_sdrtrc = Y_base + beta_val * (Y_sdrtr - Y_base)
```

The scalar beta is estimated only on the validation set:

```text
beta* = sum((Y_val - Y_base_val) * (Y_sdrtr_val - Y_base_val))
        / sum((Y_sdrtr_val - Y_base_val)^2)
```

By default beta is clipped to `[0, 1]`, so calibration can only reduce or close the residual correction, not reverse or amplify it.

## Important fixes in this version

1. `data_provider/data_factory.py`
   - `train`: `shuffle=True`
   - `val/test`: `shuffle=False`
   - `val/test`: `drop_last=args.eval_drop_last`, default `1` to match the original XLinear evaluation protocol.

2. `exp/exp_main.py`
   - `test()` saves `val_pred.npy`, `val_true.npy`, `pred.npy`, and `true.npy`.
   - Uses `np.concatenate` instead of `np.array(...).reshape(...)`, so it also supports `--eval_drop_last 0`.

3. `run_longExp.py`
   - `--is_training 0 --eval_split both` exports both validation and test predictions from an existing checkpoint.
   - `--save_val_pred 1` saves validation predictions after training.

4. `tools/calibrate_residual_beta.py`
   - Estimates beta from validation predictions.
   - Applies beta to test predictions.
   - Saves `beta.npy`, `beta_info.json`, `calibration_summary.csv`, and `calibrated_pred.npy`.

5. `tools/check_val_alignment.py`
   - Checks whether XLinear and SDR-TR validation/test targets are aligned.

## Quick Electricity workflow

Train XLinear:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/xlinear/electricity_xlinear_96_192_336_720.sh
```

Train SDR-TR:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/sdrtr/electricity_sdrtr_96_192_336_720.sh
```

If checkpoints already exist and you only need to re-export aligned validation/test predictions:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/sdrtrc/electricity_export_valtest_336_720.sh
```

Check validation alignment:

```bash
bash script/sdrtr_calibration/electricity_check_alignment_336_720.sh
```

Run SDR-TRC calibration:

```bash
bash script/sdrtr_calibration/electricity_calibrate_336_720.sh
```

## Expected alignment check

A correct export should show:

```text
base_true vs ours_true MSE: 0.0
```

or a value extremely close to zero. If this is large, do not trust beta calibration until val/test export is fixed.
