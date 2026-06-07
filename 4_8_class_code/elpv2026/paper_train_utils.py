"""LwNet / AdvEL-Net merge 训练脚本共用构建函数。"""

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_nas import ModifiedELPVDataset
from model_paper_merge import AdvELNet, Lion, LwNet
from trainer_paper_merge import MergeTrainOptions, PaperMergeTrainer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE_DIR / "modified_elpv_out2026"
DEFAULT_IMAGES_ROOT = BASE_DIR


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_classes_from_scenario(scenario):
    scenario = str(scenario)
    if scenario not in {"4", "8"}:
        raise ValueError("scenario must be '4' or '8'")
    return int(scenario)


def build_loaders(args, image_size):
    train_ds = ModifiedELPVDataset(
        out_dir=str(args.out_dir),
        images_root=str(args.images_root),
        split="train",
        scenario=str(args.scenario),
        img_size=image_size,
    )
    val_ds = ModifiedELPVDataset(
        out_dir=str(args.out_dir),
        images_root=str(args.images_root),
        split="val",
        scenario=str(args.scenario),
        img_size=image_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(model_name, num_classes):
    if model_name == "lwnet":
        return LwNet(num_classes=num_classes)
    if model_name == "advelnet":
        return AdvELNet(num_classes=num_classes)
    raise ValueError(f"Unknown model_name={model_name}")


def build_optimizer(model_name, model, args):
    if model_name == "advelnet" and args.use_lion:
        return Lion(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    classifier_lr_mult = float(getattr(args, "classifier_lr_mult", 1.0))
    if classifier_lr_mult != 1.0 and hasattr(model, "classifier"):
        classifier_params = set(map(id, model.classifier.parameters()))
        base_params = [p for p in model.parameters() if id(p) not in classifier_params]
        return torch.optim.SGD(
            [
                {"params": base_params, "lr": args.lr},
                {"params": model.classifier.parameters(), "lr": args.lr * classifier_lr_mult},
            ],
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.no_scheduler:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)


def create_components_for_model(model_name, args, image_size):
    seed_everything(args.seed)
    num_classes = num_classes_from_scenario(args.scenario)
    train_loader, val_loader = build_loaders(args, image_size)
    model1 = build_model(model_name, num_classes)
    model2 = build_model(model_name, num_classes)
    optimizer1 = build_optimizer(model_name, model1, args)
    optimizer2 = build_optimizer(model_name, model2, args)
    scheduler1 = build_scheduler(optimizer1, args)
    scheduler2 = build_scheduler(optimizer2, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    options = MergeTrainOptions(
        num_classes=num_classes,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_start_epoch=args.mixup_start_epoch,
        use_periodic_averaging=args.use_aggregation,
        avg_interval=args.avg_interval,
        avg_start_epoch=args.avg_start_epoch,
        use_ensemble_eval=args.use_ensemble_eval,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        checkpoint_dir=str(args.checkpoint_dir),
        train_second_model=getattr(args, "train_second_model", True),
    )
    trainer = PaperMergeTrainer(
        model1,
        model2,
        criterion,
        optimizer1,
        optimizer2,
        [s for s in (scheduler1, scheduler2) if s is not None],
        train_loader,
        val_loader,
        options,
        device=torch.device(args.device) if args.device != "auto" else None,
    )
    return {
        "num_classes": num_classes,
        "image_size": image_size,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model1": model1,
        "model2": model2,
        "optimizer1": optimizer1,
        "optimizer2": optimizer2,
        "criterion": criterion,
        "options": options,
        "trainer": trainer,
    }
