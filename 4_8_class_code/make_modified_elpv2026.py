#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate "modified ELPV dataset" splits from labels.csv (ELPV).
- labels.csv format (whitespace separated): image_path defect_prob cell_type
  e.g. images/cell0001.png  1.0  mono
- defect_prob in {0, 0.33333333, 0.66666667, 1}
- cell_type in {mono, poly}

Paper-aligned labeling:
- 4-class: mp-0%, mp-33%, mp-66%, mp-100%  (mono+poly merged)  :contentReference[oaicite:2]{index=2}
- 8-class: mc-0%, mc-33%, mc-66%, mc-100%, pc-0%, pc-33%, pc-66%, pc-100% :contentReference[oaicite:3]{index=3}
Split:
- train 80%, val 20%  :contentReference[oaicite:4]{index=4}
Optional:
- augmentation + balancing to target_total ~11921 (paper) :contentReference[oaicite:5]{index=5}
"""

import argparse
import os
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError as e:
    raise SystemExit("Please install Pillow: pip install pillow") from e


# -----------------------------
# Utilities
# -----------------------------

DEFECT_TO_PERCENT = {
    0.0: 0,
    1.0: 100,
    1/3: 33,
    2/3: 66,
}

def _float_to_defect_percent(x: float) -> int:
    # handle floating precision (0.33333333, 0.66666667)
    candidates = [
        (abs(x - 0.0), 0),
        (abs(x - 1.0), 100),
        (abs(x - 1/3), 33),
        (abs(x - 2/3), 66),
    ]
    candidates.sort(key=lambda t: t[0])
    if candidates[0][0] > 1e-2:
        raise ValueError(f"Unexpected defect_prob value: {x}")
    return candidates[0][1]

def _cell_to_prefix(cell_type: str) -> str:
    cell_type = str(cell_type).strip().lower()
    if cell_type == "mono":
        return "mc"  # mc-Si
    if cell_type == "poly":
        return "pc"  # pc-Si
    raise ValueError(f"Unexpected cell_type: {cell_type}")

def make_label(defect_percent: int, cell_prefix: str, scenario: str) -> str:
    """
    scenario:
      - "4": mp-{0,33,66,100}%
      - "8": {mc,pc}-{0,33,66,100}%
    """
    if scenario == "4":
        return f"mp-{defect_percent}%"
    if scenario == "8":
        return f"{cell_prefix}-{defect_percent}%"
    raise ValueError("scenario must be '4' or '8'")

def stratified_split(df: pd.DataFrame, label_col: str, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label, g in df.groupby(label_col):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_ratio))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


# -----------------------------
# Augmentation (optional)
# -----------------------------

@dataclass
class AugConfig:
    enable: bool
    images_root: str
    out_images_dir: str
    target_total: int
    seed: int
    keep_original: bool

def _safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    # ensure grayscale-like processing but keep mode flexible
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img

def _rand_int(rng: random.Random, a: int, b: int) -> int:
    return rng.randint(a, b)

def _rand_float(rng: random.Random, a: float, b: float) -> float:
    return a + (b - a) * rng.random()

def augment_one(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Basic image processing augmentations aligned with paper's examples:
    - flipping, rotation, cropping, gray level adjustments, histogram techniques, translation, etc. :contentReference[oaicite:6]{index=6}
    """
    # Work on a copy
    x = img.copy()

    # Random flip
    if rng.random() < 0.5:
        x = ImageOps.mirror(x)
    if rng.random() < 0.2:
        x = ImageOps.flip(x)

    # Random rotation (small + right-angle sometimes)
    if rng.random() < 0.3:
        angle = rng.choice([90, 180, 270])
        x = x.rotate(angle, expand=True)
    else:
        angle = _rand_float(rng, -12, 12)
        x = x.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Random translation (via padding + crop)
    if rng.random() < 0.4:
        w, h = x.size
        max_shift = int(0.06 * min(w, h))
        dx = _rand_int(rng, -max_shift, max_shift)
        dy = _rand_int(rng, -max_shift, max_shift)
        # pad then crop back
        pad = max_shift + 2
        x_pad = ImageOps.expand(x, border=pad, fill=0)
        x = x_pad.crop((pad + dx, pad + dy, pad + dx + w, pad + dy + h))

    # Random crop + resize
    if rng.random() < 0.4:
        w, h = x.size
        crop_scale = _rand_float(rng, 0.85, 0.98)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        left = _rand_int(rng, 0, max(0, w - cw))
        top = _rand_int(rng, 0, max(0, h - ch))
        x = x.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

    # Gray-level / contrast adjustments
    if rng.random() < 0.5:
        # brightness
        enhancer = ImageEnhance.Brightness(x)
        x = enhancer.enhance(_rand_float(rng, 0.85, 1.15))
    if rng.random() < 0.5:
        # contrast
        enhancer = ImageEnhance.Contrast(x)
        x = enhancer.enhance(_rand_float(rng, 0.85, 1.20))

    # Histogram technique: equalize sometimes (works best on L)
    if rng.random() < 0.25:
        if x.mode != "L":
            x_l = x.convert("L")
            x_l = ImageOps.equalize(x_l)
            x = x_l.convert(img.mode) if img.mode != "L" else x_l
        else:
            x = ImageOps.equalize(x)

    return x

def balance_and_augment(df: pd.DataFrame, label_col: str, aug: AugConfig) -> pd.DataFrame:
    """
    Create a balanced dataset (approximately equal samples per class),
    aiming for total ~ target_total (paper: ~11921). :contentReference[oaicite:7]{index=7}
    """
    if not aug.enable:
        return df

    # Determine per-class target
    labels = sorted(df[label_col].unique().tolist())
    k = len(labels)
    target_per_class = int(round(aug.target_total / k))

    rng = random.Random(aug.seed)

    os.makedirs(aug.out_images_dir, exist_ok=True)

    out_rows = []
    # optionally keep originals
    if aug.keep_original:
        for _, row in df.iterrows():
            out_rows.append(row.to_dict())

    # group by label, augment as needed
    for lab in labels:
        g = df[df[label_col] == lab]
        cur = len(g) if aug.keep_original else 0
        need = max(0, target_per_class - cur)

        if need == 0:
            continue

        # Build absolute paths
        paths = g["abs_image_path"].tolist()
        if not all(os.path.isfile(p) for p in paths):
            missing = [p for p in paths if not os.path.isfile(p)]
            raise FileNotFoundError(
                f"Augmentation enabled, but some images are missing. Example missing: {missing[0]}"
            )

        for i in range(need):
            src_abs = rng.choice(paths)
            src_rel = os.path.relpath(src_abs, aug.images_root).replace("\\", "/")

            img = _safe_open_image(src_abs)
            img_aug = augment_one(img, rng)

            # output name
            base = os.path.splitext(os.path.basename(src_rel))[0]
            out_name = f"{base}__aug_{lab.replace('%','pct').replace('-','_')}_{i:05d}.png"
            out_abs = os.path.join(aug.out_images_dir, out_name)
            img_aug.save(out_abs)

            # create new row (store as relative to dataset root for convenience)
            new_row = g.sample(n=1, random_state=rng.randint(0, 10_000_000)).iloc[0].to_dict()
            new_row["image_path"] = os.path.relpath(out_abs, os.path.dirname(aug.out_images_dir)).replace("\\", "/")
            new_row["abs_image_path"] = out_abs
            out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    # final shuffle
    out_df = out_df.sample(frac=1.0, random_state=aug.seed).reset_index(drop=True)
    return out_df


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default="labels.csv", help="Path to labels.csv (whitespace separated).")
    ap.add_argument("--scenario", type=str, choices=["4", "8"], default="8", help="4-class or 8-class scenario.")
    ap.add_argument("--images_root", type=str, default=".", help="Root dir that contains the 'images/...' paths.")
    ap.add_argument("--out_dir", type=str, default="modified_elpv_out2026", help="Output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (val = 1-train).")
    # augmentation/balancing
    ap.add_argument("--augment", default=True, help="Enable augmentation + balancing.")
    ap.add_argument("--target_total", type=int, default=11921, help="Target total samples after balancing (paper ~11921).")
    ap.add_argument("--keep_original", default=True, help="When augmenting, include original images in output CSV.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) read labels.csv (whitespace separated, no header)
    df = pd.read_csv(args.labels_csv, sep=r"\s+", header=None, names=["image_path", "defect_prob", "cell_type"])

    # 2) build labels
    df["defect_percent"] = df["defect_prob"].apply(lambda x: _float_to_defect_percent(float(x)))
    df["cell_prefix"] = df["cell_type"].apply(_cell_to_prefix)
    df["label"] = df.apply(lambda r: make_label(int(r["defect_percent"]), str(r["cell_prefix"]), args.scenario), axis=1)

    # 3) absolute image path (for augmentation / sanity checks)
    df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.abspath(os.path.join(args.images_root, p)))

    # 4) optional: augmentation + balance
    aug_cfg = AugConfig(
        enable=bool(args.augment),
        images_root=os.path.abspath(args.images_root),
        out_images_dir=os.path.join(args.out_dir, "images_aug"),
        target_total=int(args.target_total),
        seed=int(args.seed),
        keep_original=bool(args.keep_original),
    )
    if args.augment:
        df = balance_and_augment(df, label_col="label", aug=aug_cfg)

    # 5) stratified train/val split 80/20 (paper) :contentReference[oaicite:8]{index=8}
    train_df, val_df = stratified_split(df, label_col="label", train_ratio=float(args.train_ratio), seed=int(args.seed))

    # 6) save
    # Keep compact columns for training pipelines
    cols = ["image_path", "label", "defect_percent", "cell_type", "cell_prefix"]
    train_df[cols].to_csv(os.path.join(args.out_dir, f"train_{args.scenario}class.csv"), index=False)
    val_df[cols].to_csv(os.path.join(args.out_dir, f"val_{args.scenario}class.csv"), index=False)

    # Also save the full combined table for debugging
    df[cols + ["abs_image_path"]].to_csv(os.path.join(args.out_dir, f"all_{args.scenario}class_full.csv"), index=False)

    # Print summary
    def _summ(name: str, d: pd.DataFrame):
        print(f"\n[{name}] size={len(d)}")
        print(d["label"].value_counts().sort_index())

    _summ("ALL", df)
    _summ("TRAIN", train_df)
    _summ("VAL", val_df)

    print("\nDone.")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")
    print(f"- train CSV: train_{args.scenario}class.csv")
    print(f"- val   CSV: val_{args.scenario}class.csv")

if __name__ == "__main__":
    main()
