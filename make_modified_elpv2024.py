import os
import math
import argparse
import random
import shutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image, ImageOps
from skimage.io import imread
from skimage import exposure
import torchvision.transforms as T


# -------------------------
# 1) helpers: read labels.csv (space separated)
# -------------------------
def read_labels_csv(path: str):
    """
    labels.csv format (as in ELPV):
      images/cell0001.png  1.0  mono
    """
    data = np.genfromtxt(
        path,
        dtype=["|S512", "<f8", "|S32"],
        names=["path", "probability", "type"]
    )
    images = np.char.decode(data["path"])
    probs = data["probability"].astype(float)
    types = np.char.decode(data["type"])
    return images, probs, types


def prob_to_4class_name(p: float) -> str:
    """
    Original ELPV: 0, 0.33, 0.66, 1.0 (ND, PN, PD, DF)
    Paper says they annotate as ND, PN, PD, DF. :contentReference[oaicite:3]{index=3}
    """
    # robust float matching
    if abs(p - 0.0) < 1e-6:
        return "ND"
    if abs(p - (1.0/3.0)) < 1e-2 or abs(p - 0.33) < 1e-2:
        return "PN"
    if abs(p - (2.0/3.0)) < 1e-2 or abs(p - 0.66) < 1e-2:
        return "PD"
    if abs(p - 1.0) < 1e-6:
        return "DF"
    # fallback: quantize to nearest
    choices = [0.0, 1.0/3.0, 2.0/3.0, 1.0]
    nearest = min(choices, key=lambda x: abs(p - x))
    return prob_to_4class_name(nearest)


def type_to_mcpc(t: str) -> str:
    """
    In many ELPV exports: type is 'mono'/'poly'.
    Paper uses mc-Si and pc-Si. We'll map:
      mono -> mc
      poly -> pc
    """
    tl = t.strip().lower()
    if tl.startswith("mono") or tl.startswith("mc"):
        return "mc"
    if tl.startswith("poly") or tl.startswith("pc"):
        return "pc"
    # unknown -> keep as-is but still return something stable
    return tl


# -------------------------
# 2) augmentation operations described in paper 3.1
#    translation, flipping, rotation, resizing, gray adjustments,
#    histogram equalization, noise, random cropping :contentReference[oaicite:4]{index=4}
# -------------------------
def pil_hist_equalize_rgb(img: Image.Image) -> Image.Image:
    """
    Histogram equalization.
    Use skimage.exposure.equalize_hist; keep output as uint8.
    """
    arr = np.array(img)
    if arr.ndim == 2:
        eq = exposure.equalize_hist(arr)
        eq = (eq * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(eq, mode="L").convert("RGB")
    else:
        # per-channel equalize is ok for our purpose (images are grayscale-ish anyway)
        out = np.zeros_like(arr)
        for c in range(3):
            eq = exposure.equalize_hist(arr[..., c])
            out[..., c] = (eq * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


def add_gaussian_noise(img: Image.Image, sigma_range=(5.0, 20.0)) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = (arr + noise).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@dataclass
class AugConfig:
    out_size: int = 246
    do_hist_eq_prob: float = 0.30
    do_noise_prob: float = 0.30


def build_random_aug(cfg: AugConfig) -> T.Compose:
    """
    A randomized pipeline including:
      - resizing + random crop
      - translation/rotation (RandomAffine)
      - flipping
      - gray level adjustments (brightness/contrast)
    """
    # RandomResizedCrop covers resizing + random cropping
    # RandomAffine covers translation + rotation
    return T.Compose([
        T.Resize((cfg.out_size + 30, cfg.out_size + 30)),
        T.RandomResizedCrop(cfg.out_size, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomAffine(
            degrees=12,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=0.0
        ),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ])


def apply_extra_ops(img: Image.Image, cfg: AugConfig) -> Image.Image:
    if random.random() < cfg.do_hist_eq_prob:
        img = pil_hist_equalize_rgb(img)
    if random.random() < cfg.do_noise_prob:
        img = add_gaussian_noise(img)
    return img


# -------------------------
# 3) build modified dataset (8-class balanced) + optional 4-class recombined balanced
# -------------------------
def ensure_empty_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_image(img: Image.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def make_modified_dataset(
    labels_csv: str,
    images_root: str,
    out_root: str,
    target_per_8class: int,
    val_ratio: float,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    images, probs, types = read_labels_csv(labels_csv)

    # Build list: (abs_path, prob, type, class4, class8)
    samples = []
    for pth, pr, tp in zip(images, probs, types):
        class4 = prob_to_4class_name(float(pr))
        mcpc = type_to_mcpc(tp)
        class8 = f"{mcpc}-{class4}"

        # paths in csv may already include 'images/...'
        # if not, join with images_root
        if os.path.isabs(pth):
            abs_path = pth
        else:
            # if pth already startswith images/, join from cwd; else join images_root
            if os.path.exists(pth):
                abs_path = pth
            else:
                abs_path = os.path.join(images_root, pth)
        samples.append((abs_path, float(pr), tp, class4, class8))

    # group by 8-class
    by8: Dict[str, List[Tuple[str, float, str, str, str]]] = {}
    for s in samples:
        by8.setdefault(s[4], []).append(s)

    class_names_8 = sorted(by8.keys())
    if len(class_names_8) != 8:
        print("[WARN] Detected 8-class names:", class_names_8)
        print("       If your 'type' field isn't mono/poly, adjust type_to_mcpc().")

    # Prepare output dirs
    out_img_dir = os.path.join(out_root, "images_8class")
    ensure_empty_dir(out_img_dir)

    # Metadata lines to write
    meta_rows = []  # path, probability, type, class4, class8, split

    cfg = AugConfig(out_size=246)
    aug = build_random_aug(cfg)

    # For each class: copy originals + augment to target_per_8class
    for c8 in class_names_8:
        src_list = by8[c8]
        n_src = len(src_list)
        if n_src == 0:
            continue

        # 1) copy originals
        for i, (abs_path, pr, tp, c4, c8_) in enumerate(src_list):
            img = Image.fromarray(gray2rgb_safe(imread(abs_path))).convert("RGB")
            out_name = f"{c8}_orig_{i:05d}.png"
            rel_out = os.path.join("images_8class", c8, out_name)
            out_path = os.path.join(out_root, rel_out)
            save_image(img, out_path)
            meta_rows.append([rel_out, pr, tp, c4, c8, "NA"])

        # 2) augment until reaching target
        need = max(0, target_per_8class - n_src)
        for k in range(need):
            abs_path, pr, tp, c4, _ = random.choice(src_list)
            img = Image.fromarray(gray2rgb_safe(imread(abs_path))).convert("RGB")
            img = aug(img)
            img = apply_extra_ops(img, cfg)

            out_name = f"{c8}_aug_{k:05d}.png"
            rel_out = os.path.join("images_8class", c8, out_name)
            out_path = os.path.join(out_root, rel_out)
            save_image(img, out_path)
            meta_rows.append([rel_out, pr, tp, c4, c8, "NA"])

        print(f"[{c8}] source={n_src} -> total={n_src + need}")

    # Now create train/val split (stratified per 8-class) with val_ratio ~ 20% :contentReference[oaicite:5]{index=5}
    # Read back meta_rows grouped by class8
    by8_out: Dict[str, List[List[str]]] = {}
    for row in meta_rows:
        by8_out.setdefault(row[4], []).append(row)

    for c8, rows in by8_out.items():
        random.shuffle(rows)
        n_val = int(round(len(rows) * val_ratio))
        for i, row in enumerate(rows):
            row[5] = "val" if i < n_val else "train"

    # Write metadata csv
    os.makedirs(out_root, exist_ok=True)
    meta_path = os.path.join(out_root, "modified_elpv_8class.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("rel_path,probability,type,class4,class8,split\n")
        for row in meta_rows:
            f.write(",".join(map(str, row)) + "\n")

    # Also create a balanced 4-class dataset by recombining mc/pc (ND/PN/PD/DF)
    # Paper says recombine similar anomalies to get balanced 4-class (~25% each). :contentReference[oaicite:6]{index=6}
    out4_dir = os.path.join(out_root, "images_4class")
    ensure_empty_dir(out4_dir)

    # collect all generated images and regroup by class4
    by4: Dict[str, List[List[str]]] = {}
    for row in meta_rows:
        c4 = row[3]
        by4.setdefault(c4, []).append(row)

    # choose target_per_4class = 2 * target_per_8class (since 4class merges two 8-classes)
    target_per_4class = 2 * target_per_8class

    meta_rows_4 = []
    for c4, rows in by4.items():
        # rows already include train/val split from 8-class; keep same split proportion by resampling
        # First, take all existing (should be ~2*target_per_8class)
        # If more than target, downsample; if less, upsample with extra augmentation.
        random.shuffle(rows)
        selected = rows[:target_per_4class]

        # if not enough, augment from available images_8class outputs
        need = max(0, target_per_4class - len(selected))
        for k in range(need):
            base = random.choice(rows)
            src_rel = base[0]
            src_abs = os.path.join(out_root, src_rel)
            img = Image.open(src_abs).convert("RGB")
            img = aug(img)
            img = apply_extra_ops(img, cfg)

            out_name = f"{c4}_aug_{k:05d}.png"
            rel_out = os.path.join("images_4class", c4, out_name)
            out_path = os.path.join(out_root, rel_out)
            save_image(img, out_path)
            # probability/type/class8 are less meaningful in 4class; keep original
            meta_rows_4.append([rel_out, base[1], base[2], c4, base[4], base[5]])

        # copy selected originals into 4class folder
        for i, base in enumerate(selected):
            src_rel = base[0]
            src_abs = os.path.join(out_root, src_rel)
            img = Image.open(src_abs).convert("RGB")
            out_name = f"{c4}_from8_{i:05d}.png"
            rel_out = os.path.join("images_4class", c4, out_name)
            out_path = os.path.join(out_root, rel_out)
            save_image(img, out_path)
            meta_rows_4.append([rel_out, base[1], base[2], c4, base[4], base[5]])

        print(f"[4class {c4}] total={len(selected) + need}")

    meta4_path = os.path.join(out_root, "modified_elpv_4class.csv")
    with open(meta4_path, "w", encoding="utf-8") as f:
        f.write("rel_path,probability,type,class4,class8,split\n")
        for row in meta_rows_4:
            f.write(",".join(map(str, row)) + "\n")

    print("\nDone.")
    print("8-class images at:", out_img_dir)
    print("8-class csv at:", meta_path)
    print("4-class images at:", out4_dir)
    print("4-class csv at:", meta4_path)


def gray2rgb_safe(gray):
    """
    Convert grayscale numpy array to HxWx3 uint8.
    imread can return float or uint16 depending on file; normalize safely.
    """
    arr = np.array(gray)
    if arr.ndim == 3 and arr.shape[2] == 3:
        out = arr
    else:
        # make 2D
        if arr.ndim == 3:
            arr = arr[..., 0]
        # normalize to 0..255
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        out = np.stack([arr, arr, arr], axis=-1)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default="labels.csv")
    ap.add_argument("--images_root", type=str, default=".", help="root to join when paths not directly readable")
    ap.add_argument("--out_root", type=str, default="modified_elpv_out")
    ap.add_argument("--target_per_8class", type=int, default=1300,
                    help="target samples per 8-class after augmentation (paper shows ~12.5% each class) :contentReference[oaicite:7]{index=7}")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="validation ratio (paper uses 20%) :contentReference[oaicite:8]{index=8}")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_modified_dataset(
        labels_csv=args.labels_csv,
        images_root=args.images_root,
        out_root=args.out_root,
        target_per_8class=args.target_per_8class,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
