import os
import numpy as np
import pandas as pd
import torchvision as tv
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb


train_mean = [0.5, 0.5, 0.5]
train_std  = [0.5, 0.5, 0.5]
import numpy as np
import torch
from skimage.io import imread

def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """
    Make image -> HxWx3 uint8 safely.
    Handles: HxW, HxWx1, HxWx3, HxWx4, float, uint16, etc.
    """
    arr = np.asarray(img)

    # If image is 2D (H,W): replicate to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # If image is 3D (H,W,C)
    elif arr.ndim == 3:
        # HxWx1 -> HxW then replicate
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
            arr = np.stack([arr, arr, arr], axis=-1)
        # HxWx4 -> drop alpha
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # HxWx3 is ok
        elif arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel number: {arr.shape}")

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    # Convert dtype/range to uint8
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


class ModifiedELPVDataset(Dataset):
    """
    读取生成后的 modified ELPV:
      - modified_elpv_8class.csv  (class8: mc-ND ... pc-DF, 总共8类)
      - modified_elpv_4class.csv  (class4: ND/PN/PD/DF, 总共4类)

    CSV字段（由生成脚本写出）:
      rel_path,probability,type,class4,class8,split
    """

    def __init__(self,
                 csv_path: str,
                 root_dir: str,
                 mode: str = "train",
                 num_classes: int = 8,
                 return_extra_train: bool = True,
                 class_to_idx=None,type_to_idx=None):
        """
        csv_path: 例如 "modified_elpv_out/modified_elpv_8class.csv"
        root_dir: 例如 "modified_elpv_out" (用于拼接 rel_path)
        mode: "train" 或 "val" （根据csv里的 split 列过滤）
        num_classes: 8 或 4（决定用 class8 还是 class4 当 label）
        return_extra_train: True 时 train 返回 (img, label, imgtype, prob) 保持你旧代码一致
        """
        assert mode in ("train", "val")
        assert num_classes in (4, 8)
        self.mode = mode
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.return_extra_train = return_extra_train

        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((246, 246)),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.RandomRotation(degrees=10),
            tv.transforms.ColorJitter(brightness=0.1, contrast=0.1),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])

        self._transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((246, 246)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])

        df = pd.read_csv(csv_path)
        df = df[df["split"] == mode].reset_index(drop=True)

        # 选择 label 列
        label_col = "class8" if num_classes == 8 else "class4"

        if class_to_idx is None:
            classes = sorted(df[label_col].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        if type_to_idx is None:
            type_list = sorted(df["type"].astype(str).unique().tolist())
            self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        else:
            self.type_to_idx = type_to_idx

        # 建立稳定的类别映射（按字母序）
        classes = sorted(df[label_col].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # imgtype：把 type 做成 int id（mono/poly -> 0/1 或按出现顺序）
        type_list = sorted(df["type"].astype(str).unique().tolist())
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}

        self.rel_paths = df["rel_path"].astype(str).tolist()
        self.labels = [self.class_to_idx[x] for x in df[label_col].astype(str).tolist()]
        self.probs = df["probability"].astype(float).tolist()
        self.types = [self.type_to_idx[x] for x in df["type"].astype(str).tolist()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        rel = self.rel_paths[index]
        img_path = os.path.join(self.root_dir, rel)
        img_path = img_path.replace("\\", "/")
        img = imread(img_path)  # could be HxW / HxWx4 / float etc.
        rgb = to_rgb_uint8(img)  # HxWx3 uint8
        rgb_chw = torch.from_numpy(rgb).permute(2, 0, 1)  # CHW uint8

        x = self._transform_train(rgb_chw) if self.mode == "train" else self._transform_val(rgb_chw)
        y = torch.tensor(self.labels[index], dtype=torch.long)

        if self.mode == "train" and self.return_extra_train:
            imgtype = torch.tensor(self.types[index], dtype=torch.long)
            prob = torch.tensor(self.probs[index], dtype=torch.float32)
            return x, y, imgtype, prob
        else:
            return x, y
