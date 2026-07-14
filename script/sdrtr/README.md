# SDR-TR scripts

Run from the project root.

```bash
CUDA_VISIBLE_DEVICES=0 bash script/sdrtr/electricity_sdrtr_96_192.sh
CUDA_VISIBLE_DEVICES=0 bash script/sdrtr/etth1_sdrtr_96_720.sh
CUDA_VISIBLE_DEVICES=0 bash script/sdrtr/hogprice_sdrtr_ms_96_192.sh
```

The main method uses a fixed trust-region radius:

```bash
--sdr_trust_logit -4.0
--sdr_use_learnable_gate 0
```

For ablation only, set `--sdr_use_learnable_gate 1`.
