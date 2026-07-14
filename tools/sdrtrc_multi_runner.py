#!/usr/bin/env python3
"""
Batch runner for SDR-TRC multi-forecasting evaluation.

Stages:
  train_xlinear   : train XLinear and save val/test predictions
  train_sdrtr     : train SDR-TR and save val/test predictions
  export_xlinear  : load existing XLinear checkpoints and export val/test predictions
  export_sdrtr    : load existing SDR-TR checkpoints and export val/test predictions
  check           : check val/test true alignment between XLinear and SDR-TR
  calibrate       : estimate beta on validation set and apply to test set
  collect         : collect XLinear / SDR-TR / SDR-TRC metrics into one CSV

Example:
  python tools/sdrtrc_multi_runner.py --stage train_xlinear --datasets electricity,weather
  python tools/sdrtrc_multi_runner.py --stage train_sdrtr --datasets electricity,weather
  python tools/sdrtrc_multi_runner.py --stage calibrate --datasets electricity,weather
  python tools/sdrtrc_multi_runner.py --stage collect
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import numpy as np


@dataclass
class RunCfg:
    key: str
    data_path: str
    model_id_prefix: str
    data_name: str
    features: str
    target: str
    enc_in: int
    dec_in: int
    c_out: int
    horizons: List[int]
    hparams: Dict[int, Dict[str, object]]
    sdr_topk: int
    sdr_version: int = 2
    train_epochs_default: int = 30


def hp(d_model, t_ff, c_ff, c_dropout, t_dropout, head_dropout, embed_dropout,
       train_epochs=30, batch_size=32, learning_rate=1e-4, patience=3):
    return dict(
        d_model=d_model,
        t_ff=t_ff,
        c_ff=c_ff,
        c_dropout=c_dropout,
        t_dropout=t_dropout,
        head_dropout=head_dropout,
        embed_dropout=embed_dropout,
        train_epochs=train_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
    )


CFG: Dict[str, RunCfg] = {
    "etth1": RunCfg(
        key="etth1", data_path="ETTh1.csv", model_id_prefix="ETTh1", data_name="ETTh1",
        features="M", target="OT", enc_in=7, dec_in=7, c_out=7, horizons=[96, 192, 336, 720], sdr_topk=0,
        hparams={
            96: hp(128, 512, 7, 0, 0, 0.2, 0, 30, 128, 5e-4, 10),
            192: hp(128, 1024, 7, 0, 0, 0.3, 0, 30, 128, 5e-4, 5),
            336: hp(128, 1024, 7, 0, 0, 0.4, 0, 30, 128, 5e-4, 5),
            720: hp(64, 128, 7, 0, 0.1, 0.2, 0, 30, 128, 5e-4, 10),
        },
    ),
    "etth2": RunCfg(
        key="etth2", data_path="ETTh2.csv", model_id_prefix="ETTh2", data_name="ETTh2",
        features="M", target="OT", enc_in=7, dec_in=7, c_out=7, horizons=[96, 192, 336, 720], sdr_topk=0,
        hparams={
            96: hp(1024, 128, 7, 0, 0.6, 0.5, 0.2, 30, 16, 1e-4, 5),
            192: hp(1024, 128, 7, 0, 0, 0.6, 0.2, 30, 16, 1e-4, 10),
            336: hp(336, 256, 7, 0, 0, 0.4, 0, 30, 128, 1e-4, 5),
            720: hp(512, 128, 7, 0, 0, 0.3, 0, 30, 128, 1e-4, 3),
        },
    ),
    "ettm1": RunCfg(
        key="ettm1", data_path="ETTm1.csv", model_id_prefix="ETTm1", data_name="ETTm1",
        features="M", target="OT", enc_in=7, dec_in=7, c_out=7, horizons=[96, 192, 336, 720], sdr_topk=0,
        hparams={
            96: hp(512, 256, 21, 0, 0, 0.5, 0.2, 30, 32, 1e-4, 5),
            192: hp(512, 1024, 7, 0.1, 0.4, 0.6, 0.2, 30, 32, 1e-4, 3),
            336: hp(512, 512, 7, 0.1, 0.3, 0.6, 0.2, 30, 32, 2e-4, 3),
            720: hp(512, 1024, 7, 0.1, 0.4, 0.6, 0.2, 30, 32, 1e-4, 3),
        },
    ),
    "ettm2": RunCfg(
        key="ettm2", data_path="ETTm2.csv", model_id_prefix="ETTm2", data_name="ETTm2",
        features="M", target="OT", enc_in=7, dec_in=7, c_out=7, horizons=[96, 192, 336, 720], sdr_topk=0,
        hparams={
            96: hp(512, 256, 32, 0, 0, 0.6, 0.1, 20, 32, 1e-4, 3),
            192: hp(512, 128, 14, 0, 0, 0.7, 0, 20, 32, 1e-4, 3),
            336: hp(512, 256, 32, 0, 0, 0.6, 0.1, 30, 32, 1e-4, 3),
            720: hp(512, 256, 7, 0.1, 0.2, 0.4, 0.2, 15, 32, 1e-4, 3),
        },
    ),
    "weather": RunCfg(
        key="weather", data_path="weather.csv", model_id_prefix="weather", data_name="custom",
        features="M", target="OT", enc_in=21, dec_in=21, c_out=21, horizons=[96, 192, 336, 720], sdr_topk=8,
        hparams={
            96: hp(256, 256, 42, 0, 0, 0.2, 0, 20, 32, 5e-4, 10),
            192: hp(256, 128, 42, 0, 0.2, 0.3, 0, 10, 32, 5e-4, 3),
            336: hp(256, 128, 42, 0, 0.2, 0.4, 0, 10, 32, 5e-4, 3),
            720: hp(512, 256, 48, 0.2, 0, 0.4, 0.1, 30, 32, 2e-4, 10),
        },
    ),
    "electricity": RunCfg(
        key="electricity", data_path="electricity.csv", model_id_prefix="electricity", data_name="custom",
        features="M", target="OT", enc_in=321, dec_in=321, c_out=321, horizons=[96, 192, 336, 720], sdr_topk=16,
        hparams={
            96: hp(2048, 512, 32, 0.1, 0.2, 0.2, 0.2, 30, 16, 2e-4, 3),
            192: hp(2048, 512, 24, 0.1, 0.2, 0.3, 0.1, 30, 16, 1e-4, 3),
            336: hp(2048, 512, 7, 0.1, 0.2, 0.3, 0.1, 30, 16, 1e-4, 3),
            720: hp(2048, 512, 24, 0.1, 0.2, 0.6, 0.2, 30, 32, 2e-4, 3),
        },
    ),
    "traffic": RunCfg(
        key="traffic", data_path="traffic.csv", model_id_prefix="traffic", data_name="custom",
        features="M", target="OT", enc_in=862, dec_in=862, c_out=862, horizons=[96, 192, 336, 720], sdr_topk=16,
        hparams={
            96: hp(2048, 400, 64, 0.1, 0.1, 0.7, 0.1, 30, 16, 1e-4, 2),
            192: hp(2048, 336, 64, 0.1, 0.2, 0.7, 0.2, 30, 16, 1e-4, 3),
            336: hp(2048, 336, 48, 0.1, 0.3, 0.7, 0.3, 30, 16, 1e-4, 3),
            720: hp(2048, 336, 48, 0.1, 0.3, 0.7, 0.3, 30, 16, 1e-4, 3),
        },
    ),
    "hogprice": RunCfg(
        key="hogprice", data_path="hogprice260515.csv", model_id_prefix="HogPrice", data_name="custom",
        features="MS", target="hogprice", enc_in=24, dec_in=24, c_out=1, horizons=[96, 192], sdr_topk=8,
        hparams={
            96: hp(256, 128, 24, 0.1, 0.1, 0.2, 0.1, 30, 32, 2e-4, 5),
            192: hp(256, 128, 24, 0.1, 0.1, 0.2, 0.1, 30, 32, 2e-4, 5),
        },
    ),
}


def parse_list(s: str, all_values: List[str]) -> List[str]:
    if s.lower() == "all":
        return all_values
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def setting_name(cfg: RunCfg, h: int, model: str, des: str) -> str:
    model_id = f"{cfg.model_id_prefix}_96_{h}" if model == "XLinear" else f"{cfg.model_id_prefix}_SDRTR_96_{h}"
    hpv = cfg.hparams[h]
    return (
        f"{model_id}_{model}_{cfg.data_name}_ft{cfg.features}_sl96_ll48_pl{h}"
        f"_dm{hpv['d_model']}_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_{des}_0"
    )


def base_args(cfg: RunCfg, h: int, model: str, is_training: int, root_path: str, gpu: int,
              eval_split: str, eval_drop_last: int, num_workers: int) -> List[str]:
    hpv = cfg.hparams[h]
    model_id = f"{cfg.model_id_prefix}_96_{h}" if model == "XLinear" else f"{cfg.model_id_prefix}_SDRTR_96_{h}"
    des = "Exp" if model == "XLinear" else "SDR_TR"
    args = [
        "python", "-u", "run_longExp.py",
        "--random_seed", "2025",
        "--is_training", str(is_training),
        "--root_path", root_path,
        "--data_path", cfg.data_path,
        "--model_id", model_id,
        "--model", model,
        "--data", cfg.data_name,
        "--features", cfg.features,
        "--target", cfg.target,
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", str(h),
        "--enc_in", str(cfg.enc_in),
        "--dec_in", str(cfg.dec_in),
        "--c_out", str(cfg.c_out),
        "--d_model", str(hpv["d_model"]),
        "--t_ff", str(hpv["t_ff"]),
        "--c_ff", str(hpv["c_ff"]),
        "--t_dropout", str(hpv["t_dropout"]),
        "--c_dropout", str(hpv["c_dropout"]),
        "--embed_dropout", str(hpv["embed_dropout"]),
        "--head_dropout", str(hpv["head_dropout"]),
        "--des", des,
        "--train_epochs", str(hpv["train_epochs"]),
        "--patience", str(hpv["patience"]),
        "--gpu", str(gpu),
        "--num_workers", str(num_workers),
        "--itr", "1",
        "--batch_size", str(hpv["batch_size"]),
        "--learning_rate", str(hpv["learning_rate"]),
        "--eval_drop_last", str(eval_drop_last),
    ]
    if is_training == 0:
        args += ["--eval_split", eval_split]
    if model == "SDRTR":
        args += [
            "--sdr_state_dim", "128",
            "--sdr_dep_dim", "64",
            "--sdr_num_groups", "4",
            "--sdr_topk", str(cfg.sdr_topk),
            "--sdr_zero_init", "1",
            "--sdr_rng_safe_init", "1",
            "--sdr_disable_dep", "0",
            "--sdr_version", str(cfg.sdr_version),
            "--sdr_dropout", "0.05",
            "--sdr_residual_scale", "0.15",
            "--sdr_aux_lambda", "0.001",
            "--sdr_use_learnable_gate", "1",
            "--sdr_residual_target_lambda", "0.05",
            "--sdr_risk_use_quantile", "1",
            "--sdr_risk_quantile", "0.70",
            "--sdr_risk_sharpness", "8.0",
            "--sdr_risk_floor", "0.0",
            "--sdr_detach_risk", "1",
            "--sdr_debug", "1",
        ]
    return args


def run_cmd(cmd: List[str], dry_run: bool, log_path: Path = None):
    printable = " ".join(shlex.quote(x) for x in cmd)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        printable += f" | tee {shlex.quote(str(log_path))}"
    print(printable)
    if not dry_run:
        if log_path is None:
            subprocess.run(cmd, check=True)
        else:
            with open(log_path, "w", encoding="utf-8") as f:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert p.stdout is not None
                for line in p.stdout:
                    print(line, end="")
                    f.write(line)
                ret = p.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)


def metric_np(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = min(pred.shape[0], true.shape[0])
    h = min(pred.shape[1], true.shape[1])
    c = min(pred.shape[-1], true.shape[-1])
    pred = pred[:n, :h, -c:]
    true = true[:n, :h, -c:]
    return float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))


def check_alignment(base_dir: Path, ours_dir: Path, split: str = "val", tol: float = 1e-10) -> bool:
    b = np.load(base_dir / f"{split}_true.npy" if split != "test" else base_dir / "true.npy")
    o = np.load(ours_dir / f"{split}_true.npy" if split != "test" else ours_dir / "true.npy")
    mse, mae = metric_np(b, o)
    print(f"[ALIGN] {base_dir.name} vs {ours_dir.name} split={split}: true_mse={mse:.12g}, true_mae={mae:.12g}")
    return mse <= tol


def calibrate_one(cfg: RunCfg, h: int, args) -> None:
    base_dir = Path(args.results_dir) / setting_name(cfg, h, "XLinear", "Exp")
    ours_dir = Path(args.results_dir) / setting_name(cfg, h, "SDRTR", "SDR_TR")
    out_dir = Path(args.analysis_dir) / f"SDRTRC_{cfg.model_id_prefix}_{h}_{args.beta_mode}"
    if not base_dir.exists() or not ours_dir.exists():
        print(f"[SKIP] missing result dir: {base_dir} or {ours_dir}")
        return
    if args.check_alignment:
        ok_val = check_alignment(base_dir, ours_dir, "val", args.align_tol)
        ok_test = check_alignment(base_dir, ours_dir, "test", args.align_tol)
        if not (ok_val and ok_test):
            print(f"[SKIP] alignment failed for {cfg.key}-{h}")
            return
    cmd = [
        "python", "tools/calibrate_residual_beta.py",
        "--base_result_dir", str(base_dir),
        "--ours_result_dir", str(ours_dir),
        "--mode", args.beta_mode,
        "--clip_min", "0.0",
        "--clip_max", "1.0",
        "--out_dir", str(out_dir),
    ]
    run_cmd(cmd, args.dry_run)


def collect(args) -> None:
    rows = []
    for key, cfg in CFG.items():
        for h in cfg.horizons:
            base_dir = Path(args.results_dir) / setting_name(cfg, h, "XLinear", "Exp")
            ours_dir = Path(args.results_dir) / setting_name(cfg, h, "SDRTR", "SDR_TR")
            cal_dir = Path(args.analysis_dir) / f"SDRTRC_{cfg.model_id_prefix}_{h}_{args.beta_mode}"
            row = {"dataset": key, "horizon": h}
            for name, d in [("xlinear", base_dir), ("sdrtr", ours_dir)]:
                if (d / "pred.npy").exists() and (d / "true.npy").exists():
                    mse, mae = metric_np(np.load(d / "pred.npy"), np.load(d / "true.npy"))
                    row[f"{name}_mse"] = mse
                    row[f"{name}_mae"] = mae
            if (cal_dir / "calibration_summary.csv").exists():
                with open(cal_dir / "calibration_summary.csv", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r.get("split") == "test" and r.get("prediction") == "calibrated":
                            row["sdrtrc_mse"] = float(r["mse"])
                            row["sdrtrc_mae"] = float(r["mae"])
                        if r.get("split") == "val" and r.get("prediction") == "calibrated":
                            row["val_sdrtrc_mse"] = float(r["mse"])
                    info = cal_dir / "beta_info.json"
                    if info.exists():
                        with open(info, "r", encoding="utf-8") as jf:
                            beta = json.load(jf).get("beta")
                        row["beta"] = beta
            if "xlinear_mse" in row and "sdrtrc_mse" in row:
                row["rel_improve_vs_xlinear_pct"] = 100.0 * (row["sdrtrc_mse"] - row["xlinear_mse"]) / row["xlinear_mse"]
            rows.append(row)
    Path(args.analysis_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.analysis_dir) / "sdrtrc_multi_summary.csv"
    fields = sorted(set(k for r in rows for k in r.keys()), key=lambda x: ["dataset","horizon","xlinear_mse","sdrtr_mse","sdrtrc_mse","rel_improve_vs_xlinear_pct","xlinear_mae","sdrtr_mae","sdrtrc_mae","beta"].index(x) if x in ["dataset","horizon","xlinear_mse","sdrtr_mse","sdrtrc_mse","rel_improve_vs_xlinear_pct","xlinear_mae","sdrtr_mae","sdrtrc_mae","beta"] else 999)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["train_xlinear", "train_sdrtr", "export_xlinear", "export_sdrtr", "check", "calibrate", "collect"])
    parser.add_argument("--datasets", default="all", help="all or comma-separated: etth1,etth2,ettm1,ettm2,weather,electricity,traffic,hogprice")
    parser.add_argument("--horizons", default="all", help="all or comma-separated horizon list, e.g. 96,192")
    parser.add_argument("--root_path", default=os.environ.get("SEQ_ROOT", "/home/data/zsh/seq_data"))
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--analysis_dir", default="analysis_outputs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_drop_last", type=int, default=1)
    parser.add_argument("--eval_split", default="both", choices=["val", "test", "both"])
    parser.add_argument("--beta_mode", default="scalar", choices=["scalar", "group", "horizon", "channel", "group_channel"])
    parser.add_argument("--check_alignment", type=int, default=1)
    parser.add_argument("--align_tol", type=float, default=1e-10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_missing_data", type=int, default=1)
    args = parser.parse_args()

    datasets = parse_list(args.datasets, list(CFG.keys()))
    selected = []
    for k in datasets:
        if k not in CFG:
            raise KeyError(f"Unknown dataset: {k}. Available: {list(CFG.keys())}")
        cfg = CFG[k]
        data_file = Path(args.root_path) / cfg.data_path
        if args.skip_missing_data and not data_file.exists() and not args.dry_run:
            print(f"[SKIP] data file not found: {data_file}")
            continue
        selected.append(cfg)

    if args.stage == "collect":
        collect(args)
        return

    for cfg in selected:
        hs = cfg.horizons if args.horizons == "all" else [int(x) for x in args.horizons.split(",") if int(x) in cfg.horizons]
        for h in hs:
            if args.stage in ["train_xlinear", "train_sdrtr", "export_xlinear", "export_sdrtr"]:
                model = "XLinear" if "xlinear" in args.stage else "SDRTR"
                is_training = 1 if "train" in args.stage else 0
                cmd = base_args(cfg, h, model, is_training, args.root_path, args.gpu, args.eval_split, args.eval_drop_last, args.num_workers)
                log_name = f"{model}_{cfg.key}_{h}_{'train' if is_training else 'export'}.log"
                log_path = Path("logs/LongForecasting/SDRTRC_Multi") / log_name
                run_cmd(cmd, args.dry_run, log_path)
            elif args.stage == "check":
                base_dir = Path(args.results_dir) / setting_name(cfg, h, "XLinear", "Exp")
                ours_dir = Path(args.results_dir) / setting_name(cfg, h, "SDRTR", "SDR_TR")
                if base_dir.exists() and ours_dir.exists():
                    check_alignment(base_dir, ours_dir, "val", args.align_tol)
                    check_alignment(base_dir, ours_dir, "test", args.align_tol)
                else:
                    print(f"[SKIP] missing result dir for {cfg.key}-{h}")
            elif args.stage == "calibrate":
                calibrate_one(cfg, h, args)


if __name__ == "__main__":
    main()
