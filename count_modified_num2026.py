import pandas as pd
from pathlib import Path

def count_per_class(csv_path: str):
    """
    统计每个类别的样本数
    """
    df = pd.read_csv(csv_path)
    return df["label"].value_counts().sort_index()


def print_stats(train_csv: str, val_csv: str):
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)

    train_cnt = count_per_class(train_csv)
    val_cnt = count_per_class(val_csv)

    all_labels = sorted(set(train_cnt.index) | set(val_cnt.index))

    print("\n================ Dataset Statistics ================")
    print(f"{'Class':15s} {'Train':>8s} {'Val':>8s} {'Total':>8s}")
    print("-" * 50)

    for lab in all_labels:
        tr = train_cnt.get(lab, 0)
        va = val_cnt.get(lab, 0)
        print(f"{lab:15s} {tr:8d} {va:8d} {tr + va:8d}")


if __name__ == "__main__":
    # 例子：8 类
    print_stats(
        train_csv="modified_elpv_out2026/train_8class.csv",
        val_csv="modified_elpv_out2026/val_8class.csv",
    )
    print(f"\n===== 4-class statistics =====")
    # 如果是 4 类，只需改文件名
    print_stats(
        train_csv="modified_elpv_out2026/train_4class.csv",
        val_csv="modified_elpv_out2026/val_4class.csv",
    )
