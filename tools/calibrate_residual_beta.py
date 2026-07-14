import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.metrics import metric
except Exception:
    metric = None


def _load(path: str) -> np.ndarray:
    if path is None:
        raise ValueError("Expected a file path, got None")
    arr = np.load(path)
    return arr


def _align(base, ours, true):
    n = min(base.shape[0], ours.shape[0], true.shape[0])
    base = base[:n]
    ours = ours[:n]
    true = true[:n]
    h = min(base.shape[1], ours.shape[1], true.shape[1])
    c = min(base.shape[-1], ours.shape[-1], true.shape[-1])
    base = base[:, :h, -c:]
    ours = ours[:, :h, -c:]
    true = true[:, :h, -c:]
    return base, ours, true


def _mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def _mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def _full_metrics(pred, true):
    if metric is None:
        return {"mse": _mse(pred, true), "mae": _mae(pred, true)}
    mae, mse, rmse, mape, mspe, rse, corr, nse, kge, r2 = metric(pred, true)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "rse": float(rse),
        "corr": float(corr),
        "nse": float(nse),
        "kge": float(kge),
        "r2": float(r2),
    }


def _safe_beta(num, denom, clip_min, clip_max):
    beta = np.divide(num, denom, out=np.zeros_like(num, dtype=np.float64), where=np.abs(denom) > 1e-12)
    return np.clip(beta, clip_min, clip_max)


def estimate_beta(base_val, ours_val, true_val, mode="scalar", num_groups=4, clip_min=0.0, clip_max=1.0):
    correction = ours_val - base_val
    target_residual = true_val - base_val
    H = correction.shape[1]
    C = correction.shape[2]

    if mode == "scalar":
        num = np.sum(target_residual * correction, dtype=np.float64)
        den = np.sum(correction * correction, dtype=np.float64)
        beta = _safe_beta(np.asarray(num), np.asarray(den), clip_min, clip_max).item()
        beta_arr = np.asarray(beta, dtype=np.float64)
        return beta_arr

    if mode == "horizon":
        num = np.sum(target_residual * correction, axis=(0, 2), dtype=np.float64)  # [H]
        den = np.sum(correction * correction, axis=(0, 2), dtype=np.float64)
        return _safe_beta(num, den, clip_min, clip_max)

    if mode == "channel":
        num = np.sum(target_residual * correction, axis=(0, 1), dtype=np.float64)  # [C]
        den = np.sum(correction * correction, axis=(0, 1), dtype=np.float64)
        return _safe_beta(num, den, clip_min, clip_max)

    if mode == "group":
        G = max(1, min(num_groups, H))
        beta = np.zeros(G, dtype=np.float64)
        for g in range(G):
            st = (g * H) // G
            ed = ((g + 1) * H) // G
            cr = correction[:, st:ed, :]
            tr = target_residual[:, st:ed, :]
            num = np.sum(tr * cr, dtype=np.float64)
            den = np.sum(cr * cr, dtype=np.float64)
            beta[g] = _safe_beta(np.asarray(num), np.asarray(den), clip_min, clip_max).item()
        return beta

    if mode == "group_channel":
        G = max(1, min(num_groups, H))
        beta = np.zeros((G, C), dtype=np.float64)
        for g in range(G):
            st = (g * H) // G
            ed = ((g + 1) * H) // G
            cr = correction[:, st:ed, :]
            tr = target_residual[:, st:ed, :]
            num = np.sum(tr * cr, axis=(0, 1), dtype=np.float64)
            den = np.sum(cr * cr, axis=(0, 1), dtype=np.float64)
            beta[g, :] = _safe_beta(num, den, clip_min, clip_max)
        return beta

    raise ValueError(f"Unsupported beta mode: {mode}")


def apply_beta(base, ours, beta, mode="scalar", num_groups=4):
    correction = ours - base
    H = correction.shape[1]
    if mode == "scalar":
        return base + float(np.asarray(beta)) * correction

    if mode == "horizon":
        b = np.asarray(beta, dtype=np.float64).reshape(1, H, 1)
        return base + b * correction

    if mode == "channel":
        b = np.asarray(beta, dtype=np.float64).reshape(1, 1, -1)
        return base + b * correction

    if mode == "group":
        G = len(beta)
        out = base.copy()
        for g in range(G):
            st = (g * H) // G
            ed = ((g + 1) * H) // G
            out[:, st:ed, :] = base[:, st:ed, :] + float(beta[g]) * correction[:, st:ed, :]
        return out

    if mode == "group_channel":
        beta = np.asarray(beta, dtype=np.float64)
        G = beta.shape[0]
        out = base.copy()
        for g in range(G):
            st = (g * H) // G
            ed = ((g + 1) * H) // G
            out[:, st:ed, :] = base[:, st:ed, :] + beta[g].reshape(1, 1, -1) * correction[:, st:ed, :]
        return out

    raise ValueError(f"Unsupported beta mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Validation-calibrated residual beta for SDR-TR.")
    parser.add_argument("--base_result_dir", type=str, default=None,
                        help="XLinear result directory containing pred.npy/true.npy/val_pred.npy/val_true.npy")
    parser.add_argument("--ours_result_dir", type=str, default=None,
                        help="SDR-TR result directory containing pred.npy/true.npy/val_pred.npy/val_true.npy")
    parser.add_argument("--base_val_pred", type=str, default=None)
    parser.add_argument("--ours_val_pred", type=str, default=None)
    parser.add_argument("--val_true", type=str, default=None)
    parser.add_argument("--base_test_pred", type=str, default=None)
    parser.add_argument("--ours_test_pred", type=str, default=None)
    parser.add_argument("--test_true", type=str, default=None)
    parser.add_argument("--mode", type=str, default="scalar", choices=["scalar", "group", "horizon", "channel", "group_channel"])
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if args.base_result_dir is not None:
        base_dir = Path(args.base_result_dir)
        args.base_val_pred = args.base_val_pred or str(base_dir / "val_pred.npy")
        args.base_test_pred = args.base_test_pred or str(base_dir / "pred.npy")
        args.val_true = args.val_true or str(base_dir / "val_true.npy")
        args.test_true = args.test_true or str(base_dir / "true.npy")
    if args.ours_result_dir is not None:
        ours_dir = Path(args.ours_result_dir)
        args.ours_val_pred = args.ours_val_pred or str(ours_dir / "val_pred.npy")
        args.ours_test_pred = args.ours_test_pred or str(ours_dir / "pred.npy")

    base_val = _load(args.base_val_pred)
    ours_val = _load(args.ours_val_pred)
    true_val = _load(args.val_true)
    base_test = _load(args.base_test_pred)
    ours_test = _load(args.ours_test_pred)
    true_test = _load(args.test_true)

    base_val, ours_val, true_val = _align(base_val, ours_val, true_val)
    base_test, ours_test, true_test = _align(base_test, ours_test, true_test)

    beta = estimate_beta(
        base_val, ours_val, true_val,
        mode=args.mode,
        num_groups=args.num_groups,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    cal_val = apply_beta(base_val, ours_val, beta, mode=args.mode, num_groups=args.num_groups)
    cal_test = apply_beta(base_test, ours_test, beta, mode=args.mode, num_groups=args.num_groups)

    rows = []
    for split, b, o, c, t in [
        ("val", base_val, ours_val, cal_val, true_val),
        ("test", base_test, ours_test, cal_test, true_test),
    ]:
        for name, pred in [("base", b), ("ours", o), ("calibrated", c)]:
            m = _full_metrics(pred, t)
            rows.append({"split": split, "prediction": name, **m})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "beta.npy", beta)
    np.save(out_dir / "calibrated_pred.npy", cal_test)
    pd.DataFrame(rows).to_csv(out_dir / "calibration_summary.csv", index=False)

    beta_info = {
        "mode": args.mode,
        "num_groups": args.num_groups,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "beta_shape": list(np.asarray(beta).shape),
        "beta": np.asarray(beta).tolist(),
        "base_val_pred": args.base_val_pred,
        "ours_val_pred": args.ours_val_pred,
        "val_true": args.val_true,
        "base_test_pred": args.base_test_pred,
        "ours_test_pred": args.ours_test_pred,
        "test_true": args.test_true,
    }
    with open(out_dir / "beta_info.json", "w", encoding="utf-8") as f:
        json.dump(beta_info, f, indent=2, ensure_ascii=False)

    summary = pd.DataFrame(rows)
    print("Estimated beta:")
    print(np.asarray(beta))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
