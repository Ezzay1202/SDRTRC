import argparse
import os
import numpy as np


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def align_two(a, b):
    n = min(a.shape[0], b.shape[0])
    h = min(a.shape[1], b.shape[1])
    c = min(a.shape[-1], b.shape[-1])
    return a[:n, :h, -c:], b[:n, :h, -c:]


def load_result_dir(d, prefix):
    if prefix == "test":
        pred_name, true_name = "pred.npy", "true.npy"
    else:
        pred_name, true_name = f"{prefix}_pred.npy", f"{prefix}_true.npy"
    pred = np.load(os.path.join(d, pred_name))
    true = np.load(os.path.join(d, true_name))
    return pred, true


def main():
    parser = argparse.ArgumentParser(description="Check whether two result folders have aligned val/test true arrays.")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--ours_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    bp, bt = load_result_dir(args.base_dir, args.prefix)
    op, ot = load_result_dir(args.ours_dir, args.prefix)

    bp_a, bt_a = align_two(bp, bt)
    op_a, ot_a = align_two(op, ot)
    bt_b, ot_b = align_two(bt, ot)
    op_b, bt_c = align_two(op, bt)

    print(f"prefix = {args.prefix}")
    print("base pred shape:", bp_a.shape)
    print("base true shape:", bt_a.shape)
    print("ours pred shape:", op_a.shape)
    print("ours true shape:", ot_a.shape)

    print("\n[Self consistency]")
    print("base_pred vs base_true MSE:", mse(bp_a, bt_a), "MAE:", mae(bp_a, bt_a))
    print("ours_pred vs ours_true MSE:", mse(op_a, ot_a), "MAE:", mae(op_a, ot_a))

    print("\n[True alignment]")
    print("base_true vs ours_true MSE:", mse(bt_b, ot_b), "MAE:", mae(bt_b, ot_b))
    print("base_true mean/std:", float(bt_b.mean()), float(bt_b.std()))
    print("ours_true mean/std:", float(ot_b.mean()), float(ot_b.std()))

    print("\n[Cross check]")
    print("ours_pred vs base_true MSE:", mse(op_b, bt_c), "MAE:", mae(op_b, bt_c))

    if mse(bt_b, ot_b) < 1e-10:
        print("\nAlignment status: OK")
    else:
        print("\nAlignment status: NOT ALIGNED. Check data_factory.py shuffle/drop_last and re-export val/test.")


if __name__ == "__main__":
    main()
