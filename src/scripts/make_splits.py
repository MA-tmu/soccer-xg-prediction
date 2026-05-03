# src/scripts/make_splits.py

import os
import csv
import random
from typing import Dict, List, Tuple

from src.utils import load_config, ensure_dir


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _count_images(clip_dir: str) -> int:
    try:
        return sum(1 for n in os.listdir(clip_dir) if n.lower().endswith(IMAGE_EXTS))
    except Exception:
        return 0


def scan_dataset(data_root: str, frames_rgb: int, flow_file: str) -> List[Dict]:
    """
    Expect:
      data_root/
        Missed/<clip_dir>/
        Scored/<clip_dir>/
    Output rows:
      id, path, label, group
    """
    class_map = {"Missed": 0, "Scored": 1}
    samples: List[Dict] = []

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for clip_name in sorted(os.listdir(class_dir)):
            clip_dir = os.path.join(class_dir, clip_name)
            if not os.path.isdir(clip_dir):
                continue

            # basic validity
            if _count_images(clip_dir) < int(frames_rgb):
                continue
            if not os.path.isfile(os.path.join(clip_dir, flow_file)):
                # 你有的情况下要求 flow 必须存在（方便 flow 模式训练）
                continue

            sid = f"{class_name}/{clip_name}"
            group = clip_name.split("__")[0] if "__" in clip_name else clip_name

            samples.append(
                {
                    "id": sid,
                    "path": clip_dir,   # ClipDataset uses 'path'
                    "label": label,
                    "group": group,
                }
            )

    return samples


def split_by_group(samples: List[Dict], ratio: Tuple[float, float, float], seed: int):
    group_to = {}
    for s in samples:
        group_to.setdefault(s["group"], []).append(s)

    groups = list(group_to.keys())
    random.Random(seed).shuffle(groups)

    n = len(groups)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])

    train, val, test = [], [], []
    for g, items in group_to.items():
        if g in train_g:
            train.extend(items)
        elif g in val_g:
            val.extend(items)
        else:
            test.extend(items)

    return train, val, test


def write_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    fields = ["id", "path", "label", "group"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    cfg = load_config("configs/config.yaml")

    data_root = str(cfg["data_root"])
    splits_dir = str(cfg["splits_dir"])
    flow_file = str(cfg["flow_file"])
    frames_rgb = int(cfg["frames_rgb"])

    seed = int(cfg.get("split_seed", 42))
    ratio = cfg.get("split_ratio", [0.7, 0.15, 0.15])
    ratio = (float(ratio[0]), float(ratio[1]), float(ratio[2]))

    overwrite = bool(cfg.get("overwrite_splits", False))
    out_train = os.path.join(splits_dir, "train.csv")
    out_val = os.path.join(splits_dir, "val.csv")
    out_test = os.path.join(splits_dir, "test.csv")

    if (not overwrite) and (os.path.exists(out_train) or os.path.exists(out_val) or os.path.exists(out_test)):
        print("[skip] splits already exist. set overwrite_splits: true to regenerate.")
        print(" ", os.path.abspath(out_train))
        print(" ", os.path.abspath(out_val))
        print(" ", os.path.abspath(out_test))
        return

    print("[scan] data_root =", os.path.abspath(data_root))
    samples = scan_dataset(data_root=data_root, frames_rgb=frames_rgb, flow_file=flow_file)
    print("[scan] valid samples =", len(samples))
    if len(samples) == 0:
        raise RuntimeError("No valid samples found. Check data_root / frames_rgb / flow_file.")

    train, val, test = split_by_group(samples, ratio=ratio, seed=seed)
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")

    write_csv(out_train, train)
    write_csv(out_val, val)
    write_csv(out_test, test)

    print("[ok] wrote splits to:", os.path.abspath(splits_dir))


if __name__ == "__main__":
    main()
