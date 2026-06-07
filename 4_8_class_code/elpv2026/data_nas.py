import os
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.color import gray2rgb


# Precomputed for normalization (same as you used)
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]


def parse_label_string(label_str: str):
    """
    label_str examples (from my generator):
      - 4-class: "mp-0%", "mp-33%", "mp-66%", "mp-100%"
      - 8-class: "mc-0%", "pc-33%", ...
    Returns:
      defect_level(int in {0,33,66,100})
      cell_type(int): mc->0, pc->1, mp->-1
    """
    s = str(label_str).strip().lower()
    if s.endswith("%"):
        s = s[:-1]  # remove trailing %
    if "-" not in s:
        raise ValueError(f"Invalid label string: {label_str}")

    prefix, lvl = s.split("-", 1)
    try:
        defect_level = int(lvl)
    except ValueError:
        raise ValueError(f"Invalid defect level in label string: {label_str}")

    if defect_level not in (0, 33, 66, 100):
        raise ValueError(f"Unexpected defect_level={defect_level} from label_str={label_str}")

    if prefix == "mc":
        cell_type = 0
    elif prefix == "pc":
        cell_type = 1
    elif prefix == "mp":
        cell_type = -1
    else:
        raise ValueError(f"Unknown prefix={prefix} in label_str={label_str}")

    return defect_level, cell_type


class ModifiedELPVDataset(Dataset):
    """
    Loader for my generated "modified ELPV dataset" CSVs.

    Expected directory:
      out_dir/
        train_8class.csv (or train_4class.csv)
        val_8class.csv   (or val_4class.csv)
        images_aug/ ...   (optional, if augmentation enabled)

    CSV columns:
      image_path,label,defect_percent,cell_type,cell_prefix

    image_path is usually:
      - original: "images/xxx.png"  (relative to images_root)
      - augmented: "images_aug/xxx.png" (relative to out_dir, by my generator)

    Return:
      train: (img, label_id, defect_level, cell_type)
      val  : (img, label_id)
    """

    def __init__(
        self,
        out_dir: str,
        images_root: str,
        split: str = "train",
        scenario: str = "8",     # "4" or "8"
        img_size: int = 300,
        use_skimage_gray2rgb: bool = True,
    ):
        """
        Args:
            out_dir: the output folder created by generator, e.g. "modified_elpv_out"
            images_root: root that contains original "images/..." (ELPV root). For augmented images,
                         we will also try under out_dir automatically.
            split: "train" | "val"
            scenario: "4" or "8" to choose csv name train_4class.csv etc.
            img_size: model input size
        """
        super().__init__()
        self.out_dir = os.path.abspath(out_dir)
        self.images_root = os.path.abspath(images_root)
        self.split = split.lower().strip()
        self.scenario = str(scenario).strip()
        self.img_size = int(img_size)
        self.use_skimage_gray2rgb = bool(use_skimage_gray2rgb)

        if self.split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        csv_name = f"{self.split}_{self.scenario}class.csv"
        csv_path = os.path.join(self.out_dir, csv_name).replace("\\", "/")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # required columns
        for c in ["image_path", "label", "defect_percent", "cell_type", "cell_prefix"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}. Found={list(df.columns)}")

        self.df = df
        self.image_paths = df["image_path"].astype(str).values
        # ---- robust label parsing: int ids OR string labels ----
        raw_label = df["label"].astype(str).values

        # case 1) numeric ids in string form
        def _is_int_string(s: str) -> bool:
            s = s.strip()
            return s.isdigit() or (s.startswith("-") and s[1:].isdigit())

        if np.all([_is_int_string(x) for x in raw_label]):
            self.label_ids = raw_label.astype(np.int64)
            # build label_strs from defect_percent/cell_prefix as before
        else:
            # case 2) label column stores "mp-100%" etc.
            # build a stable mapping based on sorted unique labels
            uniq = sorted(set([x.strip() for x in raw_label]))
            self.class_to_idx = {c: i for i, c in enumerate(uniq)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.label_ids = np.array([self.class_to_idx[x.strip()] for x in raw_label], dtype=np.int64)

            # If your label col already has strings, we can directly use it as label_strs
            self.label_strs = np.array([x.strip() for x in raw_label], dtype=object)

        # If label_strs not set by string-label case, construct it from columns
        if not hasattr(self, "label_strs"):
            if self.scenario == "4":
                self.label_strs = np.array([f"mp-{int(x)}%" for x in df["defect_percent"].values], dtype=object)
            else:
                self.label_strs = np.array(
                    [f"{cp}-{int(dp)}%" for cp, dp in zip(df["cell_prefix"].values, df["defect_percent"].values)],
                    dtype=object
                )

        # Transforms
        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
            tv.transforms.RandomResizedCrop(
                (self.img_size, self.img_size),
                scale=(0.98, 1.0),
                ratio=(1.0, 1.0),
            ),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std),
        ])

        self._transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((self.img_size, self.img_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std),
        ])

    def __len__(self):
        return len(self.label_ids)

    def _resolve_image_path(self, rel_path: str) -> str:
        """
        Resolve image_path to absolute path.

        Priority:
        1) images_root + rel_path   (for original images/xxx.png)
        2) out_dir + rel_path       (for augmented images_aug/xxx.png saved under out_dir)
        3) if rel_path is absolute and exists, use it
        """
        rel_path = rel_path.replace("\\", "/")

        # absolute path given
        if os.path.isabs(rel_path) and os.path.exists(rel_path):
            return rel_path

        p1 = os.path.join(self.images_root, rel_path)
        if os.path.exists(p1):
            return p1

        p2 = os.path.join(self.out_dir, rel_path)
        if os.path.exists(p2):
            return p2

        raise FileNotFoundError(
            f"Image not found. Tried:\n"
            f"  1) {p1}\n"
            f"  2) {p2}\n"
            f"  3) (abs) {rel_path}\n"
        )

    def _load_as_rgb_chw_uint8(self, img_path: str) -> torch.Tensor:
        """
        Read grayscale image -> RGB tensor (C,H,W).
        Using skimage + gray2rgb to match your style.
        """
        gray_img = imread(img_path)  # HxW or HxWxC
        if gray_img.ndim == 2:
            if self.use_skimage_gray2rgb:
                rgb = gray2rgb(gray_img)  # HxWx3
            else:
                rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
        elif gray_img.ndim == 3 and gray_img.shape[2] == 3:
            rgb = gray_img
        else:
            raise ValueError(f"Unexpected image shape: {gray_img.shape} from {img_path}")

        # Ensure uint8-like range for ToPILImage
        if rgb.dtype != np.uint8:
            # if image is float in [0,1] or other, convert safely
            rgb = np.clip(rgb, 0, 255)
            if rgb.max() <= 1.5:
                rgb = (rgb * 255.0).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        # To CHW torch tensor (uint8)
        chw = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        return chw

    def __getitem__(self, index: int):
        rel_path = self.image_paths[index]
        img_path = self._resolve_image_path(rel_path)

        rgb_chw = self._load_as_rgb_chw_uint8(img_path)

        label_id = int(self.label_ids[index])
        label_str = self.label_strs[index]
        defect_level, cell_type = parse_label_string(label_str)

        if self.split == "train":
            img = self._transform_train(rgb_chw)
            return img, label_id, cell_type, defect_level
        else:
            img = self._transform_val(rgb_chw)
            return img, label_id


def build_loaders(
    out_dir: str,
    images_root: str,
    scenario: str = "8",
    img_size: int = 300,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_ds = ModifiedELPVDataset(
        out_dir=out_dir,
        images_root=images_root,
        split="train",
        scenario=scenario,
        img_size=img_size,
    )
    val_ds = ModifiedELPVDataset(
        out_dir=out_dir,
        images_root=images_root,
        split="val",
        scenario=scenario,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
