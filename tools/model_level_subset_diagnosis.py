import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rank01(x):
    return pd.Series(np.asarray(x)).rank(pct=True).values


def load_scaled_series(args):
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    data_name = str(args.data).lower()

    if data_name in ["custom", "electricity", "weather", "traffic", "exchange", "hogprice"]:
        cols = list(df_raw.columns)
        if "date" not in cols:
            raise ValueError("Custom dataset must contain a 'date' column.")
        if args.target not in cols:
            raise ValueError(f"target={args.target} not found in columns: {cols}")
        cols.remove(args.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [args.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - args.seq_len, len(df_raw) - num_test - args.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

    elif args.data in ["ETTh1", "ETTh2"]:
        border1s = [0, 12 * 30 * 24 - args.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif args.data in ["ETTm1", "ETTm2"]:
        border1s = [0, 12 * 30 * 24 * 4 - args.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        raise ValueError(f"Unsupported data type: {args.data}. Use --data custom for electricity/weather/hogprice CSVs.")

    border1, border2 = border1s[2], border2s[2]
    if args.features in ["M", "MS"]:
        df_data = df_raw[df_raw.columns[1:]]
    else:
        df_data = df_raw[[args.target]]

    scaler = StandardScaler()
    train_data = df_data.iloc[border1s[0]:border2s[0]].values
    scaler.fit(train_data)
    data = scaler.transform(df_data.values)
    return data[border1:border2]


def reconstruct_test_xy(args, n_keep=None):
    data = load_scaled_series(args)
    xs, ys = [], []
    total = len(data) - args.seq_len - args.pred_len + 1
    f_dim = -1 if args.features == "MS" else 0
    if n_keep is not None:
        total = min(total, n_keep)
    for idx in range(total):
        s_begin = idx
        s_end = s_begin + args.seq_len
        r_begin = s_end - args.label_len
        r_end = r_begin + args.label_len + args.pred_len
        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]
        xs.append(seq_x)
        ys.append(seq_y[-args.pred_len:, f_dim:])
    return np.asarray(xs), np.asarray(ys)


def sample_mse(pred, true):
    return ((pred - true) ** 2).mean(axis=(1, 2))


def sample_mae(pred, true):
    return np.abs(pred - true).mean(axis=(1, 2))


def compute_risk_scores(x, y):
    L = x.shape[1]
    hist_vol = x.std(axis=1).mean(axis=1)
    hist_range = (x.max(axis=1) - x.min(axis=1)).mean(axis=1)
    hist_shift = np.abs(x[:, L // 2:, :].mean(axis=1) - x[:, :L // 2, :].mean(axis=1)).mean(axis=1)
    fut_vol = y.std(axis=1).mean(axis=1)
    fut_range = (y.max(axis=1) - y.min(axis=1)).mean(axis=1)
    fut_trend = np.abs(y[:, -1, :] - y[:, 0, :]).mean(axis=1)
    if y.shape[1] > 2:
        dy = np.diff(y, axis=1)
        turning_proxy = (np.sign(dy[:, 1:, :]) != np.sign(dy[:, :-1, :])).mean(axis=(1, 2))
    else:
        turning_proxy = np.zeros(y.shape[0])
    hist_risk = (rank01(hist_vol) + rank01(hist_range) + rank01(hist_shift)) / 3.0
    future_hardness = (rank01(fut_vol) + rank01(fut_range) + rank01(fut_trend) + rank01(turning_proxy)) / 4.0
    composite_risk = (rank01(hist_vol) + rank01(hist_shift) + rank01(hist_range) + rank01(fut_vol) + rank01(fut_range) + rank01(fut_trend)) / 6.0
    return {
        "hist_vol": hist_vol,
        "hist_range": hist_range,
        "hist_shift": hist_shift,
        "future_vol": fut_vol,
        "future_range": fut_range,
        "future_trend": fut_trend,
        "turning_proxy": turning_proxy,
        "hist_risk": hist_risk,
        "future_hardness": future_hardness,
        "composite_risk": composite_risk,
    }


def summarize_subset(name, mask, mse_base, mse_ours, mae_base, mae_ours):
    n = int(mask.sum())
    if n == 0:
        return None
    b_mse = float(mse_base[mask].mean())
    o_mse = float(mse_ours[mask].mean())
    b_mae = float(mae_base[mask].mean())
    o_mae = float(mae_ours[mask].mean())
    return {
        "subset": name,
        "n": n,
        "xlinear_mse": b_mse,
        "ours_mse": o_mse,
        "mse_delta": o_mse - b_mse,
        "mse_delta_pct": (o_mse - b_mse) / b_mse * 100.0,
        "xlinear_mae": b_mae,
        "ours_mae": o_mae,
        "mae_delta": o_mae - b_mae,
        "mae_delta_pct": (o_mae - b_mae) / b_mae * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pred", type=str, required=True)
    parser.add_argument("--ours_pred", type=str, required=True)
    parser.add_argument("--true_path", type=str, default=None)
    parser.add_argument("--x_path", type=str, default=None)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--q", type=float, default=0.70)
    args = parser.parse_args()

    pred_base = np.load(args.base_pred)
    pred_ours = np.load(args.ours_pred)

    if args.true_path and os.path.exists(args.true_path):
        true = np.load(args.true_path)
        print(f"[INFO] Loaded true from: {args.true_path}")
    else:
        print("[WARN] true_path not provided; reconstructing true from raw data.")
        _, true = reconstruct_test_xy(args, n_keep=min(pred_base.shape[0], pred_ours.shape[0]))

    if args.x_path and os.path.exists(args.x_path):
        x = np.load(args.x_path)
        print(f"[INFO] Loaded x from: {args.x_path}")
    else:
        x, _ = reconstruct_test_xy(args, n_keep=min(pred_base.shape[0], pred_ours.shape[0], true.shape[0]))

    n = min(pred_base.shape[0], pred_ours.shape[0], true.shape[0], x.shape[0])
    pred_base, pred_ours, true, x = pred_base[:n], pred_ours[:n], true[:n], x[:n]

    if pred_base.shape != true.shape:
        print("[WARN] pred_base shape != true shape:", pred_base.shape, true.shape)
        h = min(pred_base.shape[1], true.shape[1])
        c = min(pred_base.shape[-1], true.shape[-1])
        pred_base = pred_base[:, :h, -c:]
        pred_ours = pred_ours[:, :h, -c:]
        true = true[:, :h, -c:]

    mse_base, mse_ours = sample_mse(pred_base, true), sample_mse(pred_ours, true)
    mae_base, mae_ours = sample_mae(pred_base, true), sample_mae(pred_ours, true)
    risk = compute_risk_scores(x, true)

    rows = [summarize_subset("all", np.ones(n, dtype=bool), mse_base, mse_ours, mae_base, mae_ours)]
    for k, v in risk.items():
        high_thr = np.quantile(v, args.q)
        low_thr = np.quantile(v, 1.0 - args.q)
        rows.append(summarize_subset(f"high_{k}", v >= high_thr, mse_base, mse_ours, mae_base, mae_ours))
        rows.append(summarize_subset(f"low_{k}", v <= low_thr, mse_base, mse_ours, mae_base, mae_ours))

    out = pd.DataFrame([r for r in rows if r is not None])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()
