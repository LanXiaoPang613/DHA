"""可配置双网络训练器，复用 train_merge.py 的 CE + Mixup + 周期聚合思想。"""

from dataclasses import dataclass
from pathlib import Path
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class MergeTrainOptions:
    """训练策略开关。

    use_mixup: 第二个网络是否在 warmup 后启用 mixup。
    use_periodic_averaging: 是否定期平均两个网络参数。
    use_ensemble_eval: 验证时是否融合两个网络 logits。
    """

    num_classes: int = 8
    use_mixup: bool = True
    mixup_alpha: float = 1.0
    mixup_start_epoch: int = 5
    use_periodic_averaging: bool = True
    avg_interval: int = 5
    avg_start_epoch: int = 5
    use_ensemble_eval: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    checkpoint_dir: str = "checkpoints_paper_merge"
    train_second_model: bool = True


def _json_ready(value):
    """Convert numpy / torch values to JSON-friendly Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_metrics_dict(y_true, y_pred, labels, loss):
    """Build the full metrics dict required by the paper comparison runs."""

    metrics = {
        "loss": float(loss),
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc-pure": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "p_each": precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "r_each": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_each": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }
    return _json_ready(metrics)


def mix_data_lab(x, y, alpha=1.0):
    """Mixup 数据增强，返回混合样本和两组标签。"""

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fed_avg(state_dicts):
    """两个同构网络参数平均，对应原 train_merge.py 中的 FedAvg。"""

    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged.keys():
        for other in state_dicts[1:]:
            averaged[key] += other[key]
        averaged[key] = averaged[key] / len(state_dicts)
    return averaged


class PaperMergeTrainer:
    """双模型训练器：模型 1 用 CE，模型 2 可选 Mixup，验证时可选 ensemble。"""

    def __init__(
        self,
        model1,
        model2,
        criterion,
        optimizer1,
        optimizer2,
        schedulers,
        train_loader,
        val_loader,
        options: MergeTrainOptions,
        device=None,
    ):
        self.model1 = model1
        self.model2 = model2
        self.criterion = criterion
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.schedulers = schedulers or []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.options = options
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepared_device = False

    def _prepare_device(self):
        """延迟迁移设备，方便 create_components 后在 CPU 上做 smoke test。"""

        if self._prepared_device:
            return
        self.model1.to(self.device)
        if self.options.train_second_model or self.options.use_ensemble_eval:
            self.model2.to(self.device)
        self.criterion.to(self.device)
        self._prepared_device = True

    def _train_second_model(self, x, y, epoch):
        logits2 = None
        if self.options.use_mixup and epoch >= self.options.mixup_start_epoch:
            mixed_x, y_a, y_b, lam = mix_data_lab(x, y, self.options.mixup_alpha)
            logits2 = self.model2(mixed_x)
            loss2 = mixup_criterion(self.criterion, logits2, y_a, y_b, lam)
        else:
            logits2 = self.model2(x)
            loss2 = self.criterion(logits2, y)
        return loss2

    def train_epoch(self, epoch):
        self._prepare_device()
        self.model1.train()
        if self.options.train_second_model:
            self.model2.train()
        total1, total2, total_seen = 0.0, 0.0, 0

        for batch in tqdm(self.train_loader, desc="Training"):
            x, y = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True).long()
            batch_size = x.size(0)

            self.optimizer1.zero_grad(set_to_none=True)
            logits1 = self.model1(x)
            loss1 = self.criterion(logits1, y)
            loss1.backward()
            self.optimizer1.step()

            if self.options.train_second_model:
                self.optimizer2.zero_grad(set_to_none=True)
                loss2 = self._train_second_model(x, y, epoch)
                loss2.backward()
                self.optimizer2.step()
                total2 += float(loss2.detach().item()) * batch_size

            total1 += float(loss1.detach().item()) * batch_size
            total_seen += batch_size
            if self.options.max_train_batches and total_seen >= self.options.max_train_batches * batch_size:
                break

        for scheduler in self.schedulers:
            scheduler.step()

        return total1 / max(total_seen, 1), total2 / max(total_seen, 1)

    @torch.no_grad()
    def validate(self):
        self._prepare_device()
        self.model1.eval()
        if self.options.use_ensemble_eval:
            self.model2.eval()
        total_loss, total_seen = 0.0, 0
        y_true, y_pred = [], []

        for x, y in tqdm(self.val_loader, desc="Validation"):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
            logits1 = self.model1(x)
            if self.options.use_ensemble_eval:
                logits2 = self.model2(x)
                logits = 0.5 * (logits1 + logits2)
            else:
                logits = logits1
            loss = self.criterion(logits, y)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            batch_size = x.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_seen += batch_size
            if self.options.max_val_batches and total_seen >= self.options.max_val_batches * batch_size:
                break

        labels = list(range(self.options.num_classes))
        return build_metrics_dict(y_true, y_pred, labels, total_loss / max(total_seen, 1))

    def _maybe_average(self, epoch):
        if not self.options.use_periodic_averaging:
            return
        if epoch <= self.options.avg_start_epoch:
            return
        if epoch % self.options.avg_interval != 0:
            return
        averaged = fed_avg([self.model1.state_dict(), self.model2.state_dict()])
        self.model1.load_state_dict(averaged)
        self.model2.load_state_dict(averaged)

    def fit(self, epochs):
        checkpoint_dir = Path(self.options.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(checkpoint_dir))
        metrics_jsonl = checkpoint_dir / "metrics.jsonl"
        metrics_txt = checkpoint_dir / "metrics.txt"
        history_json = checkpoint_dir / "history.json"
        best_f1 = -1.0
        best_metrics = None
        history = []

        if metrics_jsonl.exists():
            metrics_jsonl.unlink()
        if metrics_txt.exists():
            metrics_txt.unlink()

        for epoch in range(epochs):
            print(f"--- Epoch {epoch} ---")
            train_loss1, train_loss2 = self.train_epoch(epoch)
            self._maybe_average(epoch)
            metrics = self.validate()
            epoch_record = {
                "epoch": epoch,
                "train_loss1": train_loss1,
                "train_loss2": train_loss2,
                "metrics": metrics,
            }

            print("TRAIN loss:", train_loss1, train_loss2)
            print("VAL loss:", metrics["loss"])
            print("confusion matrix:", metrics["confusion"])
            print("F1 mean:", metrics["f1_weighted"], metrics["f1_each"])
            print("ACC:", metrics["acc"])
            print("BALANCE ACC:", metrics["balanced_acc"])
            print("PRECISION:", metrics["precision_weighted"], metrics["p_each"])
            print("RECALL:", metrics["recall_weighted"], metrics["r_each"])

            writer.add_scalar("Loss/train1", train_loss1, epoch)
            writer.add_scalar("Loss/train2", train_loss2, epoch)
            writer.add_scalar("Loss/val", metrics["loss"], epoch)
            writer.add_scalar("F1/val", metrics["f1_weighted"], epoch)
            writer.add_scalar("Acc/val", metrics["acc"], epoch)
            writer.add_scalar("Balanced_Acc/val", metrics["balanced_acc"], epoch)
            history.append(epoch_record)

            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            with metrics_txt.open("a", encoding="utf-8") as f:
                f.write(
                    f"epoch={epoch} train_loss1={train_loss1:.6f} train_loss2={train_loss2:.6f} "
                    f"acc={metrics['acc']:.6f} balanced_acc={metrics['balanced_acc']:.6f} "
                    f"f1_weighted={metrics['f1_weighted']:.6f}\n"
                )

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_metrics = metrics
                torch.save(
                    {
                        "state_dict1": self.model1.state_dict(),
                        "state_dict2": self.model2.state_dict(),
                        "epoch": epoch,
                        "options": self.options.__dict__,
                        "metrics": metrics,
                    },
                    checkpoint_dir / "best_checkpoint.ckp",
                )

        torch.save(
            {
                "state_dict1": self.model1.state_dict(),
                "state_dict2": self.model2.state_dict(),
                "epoch": epochs - 1,
                "options": self.options.__dict__,
                "best_f1": best_f1,
                "best_metrics": best_metrics,
            },
            checkpoint_dir / "final_checkpoint.ckp",
        )
        with history_json.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        writer.close()
        return history
