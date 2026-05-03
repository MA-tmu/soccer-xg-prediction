# src/scripts/fuse_eval.py

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

from src.metrics import compute_binary_metrics
from src.utils import ensure_dir


def load_pred_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    sid = d["sid"]
    y_true = d["y_true"].astype(np.float32)
    y_prob = d["y_prob"].astype(np.float32)
    return sid, y_true, y_prob


def build_index(sid: np.ndarray) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i in range(len(sid)):
        idx[str(sid[i])] = i
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="rgb .npz from eval.py")
    parser.add_argument("--flow", type=str, required=True, help="flow .npz from eval.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None, help="output fused npz (default beside rgb npz)")
    args = parser.parse_args()

    sid_r, y_r, p_r = load_pred_npz(args.rgb)
    sid_f, y_f, p_f = load_pred_npz(args.flow)

    idx_f = build_index(sid_f)

    fused_sid = []
    fused_y = []
    fused_p = []

    miss = 0
    for i in range(len(sid_r)):
        sid = str(sid_r[i])
        if sid not in idx_f:
            miss += 1
            continue
        j = idx_f[sid]

        # sanity: labels must match
        if float(y_r[i]) != float(y_f[j]):
            raise RuntimeError(f"Label mismatch for sid={sid}: rgb={y_r[i]} flow={y_f[j]}")

        pr = float(p_r[i])
        pf = float(p_f[j])
        fused = 0.5 * (pr + pf)

        fused_sid.append(sid)
        fused_y.append(float(y_r[i]))
        fused_p.append(float(fused))

    if len(fused_sid) == 0:
        raise RuntimeError("No matched samples between rgb and flow predictions.")

    y_true = np.asarray(fused_y, dtype=np.float32)
    y_prob = np.asarray(fused_p, dtype=np.float32)

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(args.threshold))

    # output paths
    base_dir = os.path.dirname(os.path.abspath(args.rgb))
    ensure_dir(base_dir)

    if args.out is not None:
        out_path = args.out
    else:
        out_path = os.path.join(base_dir, "fused_pred.npz")

    np.savez_compressed(
        out_path,
        sid=np.asarray(fused_sid, dtype=object),
        y_true=y_true,
        y_prob=y_prob,
    )

    metrics_out = {"matched": int(len(fused_sid)), "miss_in_flow": int(miss), "threshold": float(args.threshold), **metrics}
    metrics_path = os.path.join(base_dir, "fused_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    # 打印你要的顺序
    keys = ["acc", "precision", "recall", "f1", "balanced_acc", "auc", "ap", "mcc", "brier", "logloss", "mae"]
    parts = [f"matched={len(fused_sid)}", f"miss_in_flow={miss}"]
    for k in keys:
        parts.append(f"{k}={float(metrics.get(k, 0.0)):.4f}")
    print(" | ".join(parts))

    print(f"[saved] {out_path}")
    print(f"[saved] {metrics_path}")


if __name__ == "__main__":
    main()