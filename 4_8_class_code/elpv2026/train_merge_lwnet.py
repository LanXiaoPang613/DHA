"""使用 LwNet 替换 train_merge.py 原模型的双网络训练入口。"""

import argparse

from paper_train_utils import DEFAULT_IMAGES_ROOT, DEFAULT_OUT_DIR, create_components_for_model


def _add_bool_pair(parser, positive, negative, dest, default, help_text):
    parser.add_argument(positive, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(negative, dest=dest, action="store_false")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train LwNet with merge-style dual-network strategy.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--images-root", type=str, default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--scenario", choices=("4", "8"), default="8")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--milestones", type=int, nargs="*", default=[100, 200])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--no-scheduler", action="store_true", default=False)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-start-epoch", type=int, default=5)
    parser.add_argument("--avg-interval", type=int, default=5)
    parser.add_argument("--avg-start-epoch", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_lwnet_merge")
    parser.add_argument("--classifier-lr-mult", type=float, default=1.0)
    _add_bool_pair(parser, "--use-mixup", "--no-mixup", "use_mixup", True, "enable second-network mixup")
    _add_bool_pair(parser, "--use-aggregation", "--no-aggregation", "use_aggregation", True, "enable periodic parameter averaging")
    _add_bool_pair(parser, "--ensemble-eval", "--single-eval", "use_ensemble_eval", True, "ensemble two models at validation")
    _add_bool_pair(parser, "--train-second-model", "--single-model-train", "train_second_model", True, "train the second network branch")
    parser.set_defaults(use_lion=False, beta1=0.9, beta2=0.99)
    return parser


def create_components(args):
    return create_components_for_model("lwnet", args, image_size=246)


def main():
    args = build_arg_parser().parse_args()
    components = create_components(args)
    print("LwNet merge training")
    print("dataset:", args.out_dir)
    print("scenario:", args.scenario, "image_size:", components["image_size"])
    print("switches:", components["options"])
    components["trainer"].fit(args.epochs)


if __name__ == "__main__":
    main()
