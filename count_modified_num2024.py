import csv
from collections import defaultdict

def count_samples_per_class(csv_path, class_field="class8"):
    """
    统计每个类别在 train / val 中的样本数
    """
    stats = defaultdict(lambda: defaultdict(int))

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row[class_field]
            split = row["split"]
            stats[cls][split] += 1

    return stats


def pretty_print(stats, title):
    print(f"\n===== {title} =====")
    print(f"{'Class':15s} {'Train':>8s} {'Val':>8s} {'Total':>8s}")
    print("-" * 45)
    for cls, cnt in sorted(stats.items()):
        train_n = cnt.get("train", 0)
        val_n = cnt.get("val", 0)
        print(f"{cls:15s} {train_n:8d} {val_n:8d} {train_n + val_n:8d}")


if __name__ == "__main__":
    csv_8class = "modified_elpv_out2024/modified_elpv_8class.csv"
    stats_8 = count_samples_per_class(csv_8class, class_field="class8")
    pretty_print(stats_8, "8-class dataset")

    csv_4class = "modified_elpv_out2024/modified_elpv_4class.csv"
    stats4 = count_samples_per_class(csv_4class, class_field="class4")
    pretty_print(stats4, "4-class dataset")
