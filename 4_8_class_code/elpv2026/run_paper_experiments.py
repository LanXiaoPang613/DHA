"""Run LwNet / AdvEL-Net experiments on modified_elpv_out2026.

The script creates one isolated output directory for every combination:
model x hyperparameter profile x 4/8-class scenario.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from paper_train_utils import DEFAULT_IMAGES_ROOT, DEFAULT_OUT_DIR, create_components_for_model


@dataclass(frozen=True)
class ExperimentItem:
    model: str
    profile: str
    scenario: str
    image_size: int
    output_dir: str
    settings: dict


class Tee:
    """Write stdout/stderr to both terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _paper_profile(model):
    if model == "lwnet":
        return {
            "epochs": 50,
            "batch_size": 64,
            "lr": 1e-4,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "milestones": [100, 200],
            "gamma": 0.1,
            "no_scheduler": True,
            "use_lion": False,
            "beta1": 0.9,
            "beta2": 0.99,
            "classifier_lr_mult": 10.0,
            "use_mixup": False,
            "mixup_alpha": 1.0,
            "mixup_start_epoch": 5,
            "use_aggregation": False,
            "avg_interval": 5,
            "avg_start_epoch": 5,
            "use_ensemble_eval": False,
            "train_second_model": False,
        }
    if model == "advelnet":
        return {
            "epochs": 30,
            "batch_size": 16,
            "lr": 1e-4,
            "momentum": 0.9,
            "weight_decay": 0.1,
            "label_smoothing": 0.05,
            "milestones": [100, 200],
            "gamma": 0.1,
            "no_scheduler": True,
            "use_lion": True,
            "beta1": 0.9,
            "beta2": 0.99,
            "classifier_lr_mult": 1.0,
            "use_mixup": False,
            "mixup_alpha": 1.0,
            "mixup_start_epoch": 5,
            "use_aggregation": False,
            "avg_interval": 5,
            "avg_start_epoch": 5,
            "use_ensemble_eval": False,
            "train_second_model": False,
        }
    raise ValueError(f"Unsupported model: {model}")


def _local_merge_profile(model):
    return {
        "epochs": 200,
        "batch_size": 32,
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 3.0e-5,
        "label_smoothing": 0.0,
        "milestones": [100, 200],
        "gamma": 0.1,
        "no_scheduler": False,
        "use_lion": False,
        "beta1": 0.9,
        "beta2": 0.99,
        "classifier_lr_mult": 1.0,
        "use_mixup": True,
        "mixup_alpha": 1.0,
        "mixup_start_epoch": 5,
        "use_aggregation": True,
        "avg_interval": 5,
        "avg_start_epoch": 5,
        "use_ensemble_eval": True,
        "train_second_model": True,
    }


def _profile_settings(model, profile):
    if profile == "paper":
        return _paper_profile(model)
    if profile == "local_merge":
        return _local_merge_profile(model)
    raise ValueError(f"Unsupported profile: {profile}")


def _image_size(model):
    if model == "lwnet":
        return 246
    if model == "advelnet":
        return 224
    raise ValueError(f"Unsupported model: {model}")


def build_experiment_plan(models, profiles, scenarios, base_output_dir="paper_model_runs"):
    plan = []
    for model in models:
        for profile in profiles:
            for scenario in scenarios:
                output_dir = Path(base_output_dir) / model / profile / f"{scenario}class"
                plan.append(
                    ExperimentItem(
                        model=model,
                        profile=profile,
                        scenario=str(scenario),
                        image_size=_image_size(model),
                        output_dir=str(output_dir),
                        settings=_profile_settings(model, profile),
                    )
                )
    return plan


def _namespace_for_item(item, cli_args):
    settings = dict(item.settings)
    if cli_args.epochs_override is not None:
        settings["epochs"] = cli_args.epochs_override
    if cli_args.max_train_batches is not None:
        settings["max_train_batches"] = cli_args.max_train_batches
    else:
        settings["max_train_batches"] = None
    if cli_args.max_val_batches is not None:
        settings["max_val_batches"] = cli_args.max_val_batches
    else:
        settings["max_val_batches"] = None

    return SimpleNamespace(
        out_dir=str(cli_args.out_dir),
        images_root=str(cli_args.images_root),
        scenario=item.scenario,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        seed=cli_args.seed,
        device=cli_args.device,
        checkpoint_dir=item.output_dir,
        **settings,
    )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_one_experiment(item, cli_args):
    output_dir = Path(item.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _namespace_for_item(item, cli_args)
    config = {
        "model": item.model,
        "profile": item.profile,
        "scenario": item.scenario,
        "image_size": item.image_size,
        "args": vars(args),
    }
    _write_json(output_dir / "config.json", config)

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
            print("=" * 80)
            print(f"model={item.model} profile={item.profile} scenario={item.scenario} image_size={item.image_size}")
            print(f"output_dir={output_dir}")
            print("settings:", item.settings)
            components = create_components_for_model(item.model, args, image_size=item.image_size)
            history = components["trainer"].fit(args.epochs)
            best_record = max(history, key=lambda record: record["metrics"]["f1_weighted"]) if history else None
            summary = {
                "model": item.model,
                "profile": item.profile,
                "scenario": item.scenario,
                "best": best_record,
                "epochs": args.epochs,
            }
            _write_json(output_dir / "summary.json", summary)
            print("summary:", summary)
            return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run paper and local-merge profiles for LwNet and AdvEL-Net.")
    parser.add_argument("--models", nargs="+", choices=("lwnet", "advelnet"), default=["lwnet", "advelnet"])
    parser.add_argument("--profiles", nargs="+", choices=("paper", "local_merge"), default=["paper", "local_merge"])
    parser.add_argument("--scenarios", nargs="+", choices=("4", "8"), default=["4", "8"])
    parser.add_argument("--base-output-dir", default="paper_model_runs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    return parser


def main():
    cli_args = build_arg_parser().parse_args()
    plan = build_experiment_plan(cli_args.models, cli_args.profiles, cli_args.scenarios, cli_args.base_output_dir)
    summaries = []
    summary_jsonl = Path(cli_args.base_output_dir) / "summary.jsonl"
    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if summary_jsonl.exists() and not cli_args.skip_existing:
        summary_jsonl.unlink()

    for item in plan:
        final_checkpoint = Path(item.output_dir) / "final_checkpoint.ckp"
        if cli_args.skip_existing and final_checkpoint.exists():
            print(f"skip_existing={item.output_dir}")
            continue
        summary = run_one_experiment(item, cli_args)
        summaries.append(summary)
        with summary_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    _write_json(Path(cli_args.base_output_dir) / "summary_all.json", summaries)


if __name__ == "__main__":
    main()
