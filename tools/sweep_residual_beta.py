import argparse
import os
import numpy as np
import pandas as pd


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def align(base, ours, true):
    n = min(base.shape[0], ours.shape[0], true.shape[0])
    h = min(base.shape[1], ours.shape[1], true.shape[1])
    c = min(base.shape[-1], ours.shape[-1], true.shape[-1])
    return base[:n, :h, -c:], ours[:n, :h, -c:], true[:n, :h, -c:]


def main():
    parser = argparse.ArgumentParser(description="Post-hoc beta sweep for diagnosis only. Do not use test beta as final result.")
    parser.add_argument("--base_pred", type=str, required=True)
    parser.add_argument("--ours_pred", type=str, required=True)
    parser.add_argument("--true_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--beta_min", type=float, default=-1.0)
    parser.add_argument("--beta_max", type=float, default=2.0)
    parser.add_argument("--beta_step", type=float, default=0.05)
    args = parser.parse_args()

    base = np.load(args.base_pred)
    ours = np.load(args.ours_pred)
    true = np.load(args.true_path)
    base, ours, true = align(base, ours, true)
    correction = ours - base

    rows = []
    beta = args.beta_min
    while beta <= args.beta_max + 1e-9:
        pred = base + beta * correction
        rows.append({"beta": beta, "mse": mse(pred, true), "mae": mae(pred, true)})
        beta += args.beta_step

    df = pd.DataFrame(rows)
    best = df.loc[df["mse"].idxmin()]
    base_mse = mse(base, true)
    ours_mse = mse(ours, true)

    print("Base/XLinear MSE:", base_mse)
    print("Ours/SDR-TR MSE:", ours_mse)
    print("Best beta:", best["beta"])
    print("Best MSE:", best["mse"])
    print("Best MAE:", best["mae"])

    denom = np.sum(correction * correction)
    if denom > 1e-12:
        beta_star = np.sum((true - base) * correction) / denom
        pred_star = base + beta_star * correction
        print("Closed-form beta*:", float(beta_star))
        print("Closed-form beta* MSE:", mse(pred_star, true))
        print("Closed-form beta* MAE:", mae(pred_star, true))

    if args.out_csv is not None:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
