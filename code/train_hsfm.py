"""
HSFM: Hard-Set-Guided Feature-Space Meta-Learning for Robust Classification
under Spurious Correlations (ECCV 2026).

Given a frozen ERM backbone, HSFM optimizes a class-balanced support set of
learnable feature embeddings H so that a linear head adapted on H achieves lower
loss on hard validation examples (top-K highest loss per class). See Algorithm 1
in the paper for the full training procedure.
"""

from __future__ import annotations

import argparse
import datetime
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as tv_models
from dominoes import Dominoes


# Paper Table 11: ResNet-50 hyperparameters per spurious-correlation benchmark.
@dataclass(frozen=True)
class DatasetConfig:
    support_size_per_class: int
    T: int
    inner_lr: float
    outer_lr: float
    KH: int
    K_hard: int
    epochs: int


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "celeba": DatasetConfig(1024, 10, 5e-5, 1.0, 5, 256, 20),
    "waterbirds": DatasetConfig(16, 15, 5e-5, 1.0, 15, 64, 40),
    "metashift": DatasetConfig(32, 15, 5e-5, 1e-2, 10, 64, 10),
    "dominoes": DatasetConfig(16, 10, 1e-4, 1.0, 15, 256, 40),
}


def preferred_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_repo_cache_env() -> None:
    """Use writable cache directories under this repo when env vars are missing."""
    repo = Path(__file__).resolve().parent
    targets = {
        "TORCH_HOME": repo / "cache" / "torch",
        "HF_HOME": repo / "cache" / "huggingface",
        "XDG_CACHE_HOME": repo / "cache" / "xdg",
    }

    def broken(val: str) -> bool:
        v = val.strip()
        if not v:
            return True
        if v in ("/cache", "/logs"):
            return True
        return v.startswith("/cache/") or v.startswith("/logs/")

    for key, path in targets.items():
        path.mkdir(parents=True, exist_ok=True)
        if broken(os.environ.get(key, "")):
            os.environ[key] = str(path)


class ERMResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def forward_linear(x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return x @ params["weight"].t() + params["bias"]


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 4:
        img, _img2, label_onehot, env_onehot = batch
    elif len(batch) == 3:
        img, label_onehot, env_onehot = batch
    else:
        raise RuntimeError(f"Unexpected batch format with len={len(batch)}")
    labels = label_onehot.argmax(dim=1)
    envs = env_onehot.argmax(dim=1)
    return img, labels, envs


def select_hard_set(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    params: dict[str, torch.Tensor],
    k_per_class: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K highest-loss validation samples per class (Eq. 5-6)."""
    embeds = embeds.to(device)
    labels = labels.to(device)
    logits = forward_linear(embeds, params)
    losses = nn.CrossEntropyLoss(reduction="none")(logits, labels)

    hard_indices = []
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_losses = losses[class_mask]
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        if class_losses.numel() == 0:
            continue
        k = min(int(k_per_class), int(class_losses.numel()))
        topk = torch.topk(class_losses, k=k, largest=True).indices
        hard_indices.append(class_indices[topk])

    if not hard_indices:
        raise RuntimeError("No hard samples found in the validation set.")
    hard_indices = torch.cat(hard_indices, dim=0)
    return embeds[hard_indices], labels[hard_indices]


@torch.no_grad()
def eval_worst_group_accuracy(
    feats: torch.Tensor,
    labels: torch.Tensor,
    envs: torch.Tensor,
    params: dict[str, torch.Tensor],
    device: torch.device,
    n_groups: int | None = None,
) -> Tuple[float, float, list[float]]:
    """Return average accuracy, worst-group accuracy, and per-group accuracies."""
    x = feats.to(device)
    y = labels.to(device)
    e = envs.to(device)
    logits = forward_linear(x, params)
    preds = logits.argmax(dim=1)

    avg_acc = 100.0 * float((preds == y).float().mean().item())
    inferred_n_groups = int(e.max().item()) + 1 if e.numel() > 0 else 0
    use_n_groups = int(n_groups) if n_groups is not None else inferred_n_groups
    group_correct = [0 for _ in range(use_n_groups)]
    group_total = [0 for _ in range(use_n_groups)]
    for i in range(y.shape[0]):
        g = int(e[i].item())
        if 0 <= g < use_n_groups:
            group_total[g] += 1
            if int(preds[i].item()) == int(y[i].item()):
                group_correct[g] += 1
    group_acc = [
        100.0 * group_correct[i] / group_total[i] if group_total[i] > 0 else 0.0
        for i in range(use_n_groups)
    ]
    worst_group_acc = min(group_acc) if group_acc else 0.0
    return avg_acc, worst_group_acc, group_acc


@torch.no_grad()
def compute_features_for_indices(
    dataset,
    indices: list[int],
    backbone: nn.Module,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    backbone.eval()
    feats = []
    for i in range(0, len(indices), int(batch_size)):
        batch_idx = indices[i : i + int(batch_size)]
        imgs = []
        for idx in batch_idx:
            sample = dataset[int(idx)]
            imgs.append(sample[0])
        img_batch = torch.stack(imgs, dim=0).to(device)
        f = backbone(img_batch).flatten(1).detach()
        feats.append(f)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def compute_all_features(
    loader: DataLoader,
    backbone: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backbone.eval()
    feats_all = []
    labels_all = []
    envs_all = []
    for batch in loader:
        img, labels, envs = unpack_batch(batch)
        img = img.to(device)
        f = backbone(img).flatten(1).detach().cpu()
        feats_all.append(f)
        labels_all.append(labels.detach().cpu())
        envs_all.append(envs.detach().cpu())
    return torch.cat(feats_all, dim=0), torch.cat(labels_all, dim=0), torch.cat(envs_all, dim=0)


def build_backbone(model: ERMResNet50) -> nn.Module:
    return nn.Sequential(*list(model.model.children())[:-1])


def extract_model_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        return ckpt_obj
    raise RuntimeError("Checkpoint must be a state_dict or a dict containing 'model_state_dict'.")


def format_group_values(values: list[float | int]) -> str:
    return "/".join([f"{v:.1f}" if isinstance(v, float) else str(v) for v in values])


class TeeIO:
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


def load_datasets(args):
    if args.dataset == "waterbirds":
        from Waterbirds import get_transform_cub, get_waterbird_dataset

        t_eval = get_transform_cub(False)
        train_dataset = get_waterbird_dataset("train", t_eval, t_eval, root_dir=args.waterbirds_root_dir)
        val_dataset = get_waterbird_dataset("val", t_eval, t_eval, root_dir=args.waterbirds_root_dir)
        test_dataset = get_waterbird_dataset("test", t_eval, t_eval, root_dir=args.waterbirds_root_dir)
    elif args.dataset == "celeba":
        from CelebA import CelebADataset

        t_eval = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = CelebADataset("train", root_dir=args.celeba_root_dir, transform=t_eval)
        val_dataset = CelebADataset("val", root_dir=args.celeba_root_dir, transform=t_eval)
        test_dataset = CelebADataset("test", root_dir=args.celeba_root_dir, transform=t_eval)
    elif args.dataset == "dominoes":
        train_dataset = Dominoes("train", path=args.dominoes_data_dir, transform=None)
        val_dataset = Dominoes("val", path=args.dominoes_data_dir, transform=None)
        test_dataset = Dominoes("test", path=args.dominoes_data_dir, transform=None)
    else:
        from Metashift import MetaDatasetCatDog, get_transform_metashift

        metashift = MetaDatasetCatDog(root_dir=args.metashift_data_dir)
        t_eval = get_transform_metashift(False)
        metashift.train_transform = t_eval
        metashift.eval_transform = t_eval
        splits = metashift.get_splits(["train", "val", "test"], train_frac=1.0)
        train_dataset = splits["train"]
        val_dataset = splits["val"]
        test_dataset = splits["test"]

    return train_dataset, val_dataset, test_dataset


def get_train_labels(train_dataset) -> list[int]:
    train_labels = getattr(train_dataset, "y_array", None)
    if train_labels is None and hasattr(train_dataset, "get_label_array"):
        train_labels = train_dataset.get_label_array()
    if train_labels is not None:
        return list(train_labels)

    labels_list = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        label_onehot = sample[2] if len(sample) == 4 else sample[1]
        labels_list.append(int(label_onehot.argmax().item()))
    return labels_list


def inner_loop_adapt_head(
    support_emb: torch.Tensor,
    support_labels: torch.Tensor,
    theta_0: dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
) -> dict[str, torch.Tensor]:
    """Adapt linear head on support embeddings for T steps (Eq. 12-14)."""
    theta = {k: v.detach().clone().requires_grad_(True) for k, v in theta_0.items()}
    for _ in range(int(inner_steps)):
        logits = forward_linear(support_emb, theta)
        loss_in = nn.CrossEntropyLoss()(logits, support_labels)
        grads = torch.autograd.grad(loss_in, list(theta.values()), create_graph=True)
        theta = {k: p - float(inner_lr) * g for (k, p), g in zip(theta.items(), grads)}
    return theta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HSFM: Hard-Set-Guided Feature-Space Meta-Learning (ResNet-50)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="waterbirds",
        choices=["waterbirds", "celeba", "dominoes", "metashift"],
    )
    parser.add_argument("--waterbirds_root_dir", type=str, default="../DaC/waterbird_complete95_forest2water2")
    parser.add_argument("--celeba_root_dir", type=str, default="../DaC/celeba")
    parser.add_argument("--dominoes_data_dir", type=str, default="Dominoes_SP90 2")
    parser.add_argument("--metashift_data_dir", type=str, default="metashifts/MetaDatasetCatDog")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_checkpoint", type=str, default="checkpoints/waterbirds_erm.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--support_size_per_class", type=int, default=None, help="Support set size per class.")
    parser.add_argument("--T", type=int, default=None, help="Inner-loop and head update steps.")
    parser.add_argument("--inner_lr", type=float, default=None, help="Inner-loop learning rate alpha.")
    parser.add_argument("--outer_lr", type=float, default=None, help="Meta learning rate eta.")
    parser.add_argument("--KH", type=int, default=None, help="Support embedding updates per epoch.")
    parser.add_argument("--K_hard", type=int, default=None, help="Hard samples per class from validation.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of HSFM training epochs.")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--save_path", type=str, default="outputs/hsfm_final.pt")
    parser.add_argument("--best_save_path", type=str, default="outputs/hsfm_best.pt")
    parser.add_argument("--cache_root_dir", type=str, default="cache")
    parser.add_argument("--val_feature_cache", type=str, default=None)
    parser.add_argument("--refresh_val_cache", action="store_true")
    parser.add_argument("--test_feature_cache", type=str, default=None)
    parser.add_argument("--refresh_test_cache", action="store_true")
    parser.add_argument(
        "--save_repr_checkpoints",
        action="store_true",
        help="Save before/after support embeddings for visualization (Sec. 5.3).",
    )
    parser.add_argument("--repr_checkpoint_root", type=str, default="outputs/repr_checkpoints")
    return parser.parse_args()


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    if args.support_size_per_class is None:
        args.support_size_per_class = cfg.support_size_per_class
    if args.T is None:
        args.T = cfg.T
    if args.inner_lr is None:
        args.inner_lr = cfg.inner_lr
    if args.outer_lr is None:
        args.outer_lr = cfg.outer_lr
    if args.KH is None:
        args.KH = cfg.KH
    if args.K_hard is None:
        args.K_hard = cfg.K_hard
    if args.epochs is None:
        args.epochs = cfg.epochs


def main() -> None:
    ensure_repo_cache_env()
    args = parse_args()
    apply_dataset_defaults(args)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hsfm_{args.dataset}_{ts}.log"
    log_fh = open(log_file, "a", encoding="utf-8")
    sys.stdout = TeeIO(sys.__stdout__, log_fh)
    sys.stderr = TeeIO(sys.__stderr__, log_fh)

    print(f"Logging to: {log_file}")
    print("HSFM training configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value}")

    torch.manual_seed(int(args.seed))
    device = preferred_device()
    print(f"Device: {device}")

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4)

    model = ERMResNet50().to(device)
    ckpt_path = Path(args.model_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"ERM checkpoint not found: {ckpt_path}. "
            "Train an ERM ResNet-50 on the target dataset first."
        )
    print(f"Loading ERM checkpoint from {ckpt_path}")
    ckpt_obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(extract_model_state_dict(ckpt_obj))
    backbone = build_backbone(model).to(device)

    dataset_cache_name = {
        "dominoes": "Dominoes",
        "celeba": "CelebA",
        "waterbirds": "Waterbirds",
        "metashift": "MetaShift",
    }[args.dataset]
    cache_dir = Path(args.cache_root_dir) / dataset_cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Feature cache dir: {cache_dir}")

    val_cache_path = Path(args.val_feature_cache) if args.val_feature_cache else cache_dir / "val_features.pt"
    if val_cache_path.exists() and not args.refresh_val_cache:
        val_feat_cache = torch.load(str(val_cache_path), map_location="cpu")
        val_feats = val_feat_cache["features"]
        val_labels = val_feat_cache["labels"]
        val_envs = val_feat_cache["envs"]
    else:
        print(f"Building validation feature cache: {val_cache_path}")
        val_feats, val_labels, val_envs = compute_all_features(val_loader, backbone, device)
        torch.save({"features": val_feats, "labels": val_labels, "envs": val_envs}, str(val_cache_path))

    test_cache_path = Path(args.test_feature_cache) if args.test_feature_cache else cache_dir / "test_features.pt"
    if test_cache_path.exists() and not args.refresh_test_cache:
        test_feat_cache = torch.load(str(test_cache_path), map_location="cpu")
        test_feats = test_feat_cache["features"]
        test_labels = test_feat_cache["labels"]
        test_envs = test_feat_cache["envs"]
    else:
        print(f"Building test feature cache: {test_cache_path}")
        test_feats, test_labels, test_envs = compute_all_features(test_loader, backbone, device)
        torch.save({"features": test_feats, "labels": test_labels, "envs": test_envs}, str(test_cache_path))

    eval_n_groups = 2 if args.dataset == "metashift" else 4
    train_labels = get_train_labels(train_dataset)
    class0_idx = torch.where(torch.tensor(train_labels) == 0)[0]
    class1_idx = torch.where(torch.tensor(train_labels) == 1)[0]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed))

    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_feats = backbone(sample_batch[0].to(device)).flatten(1)
        embed_dim = int(sample_feats.shape[1])
        print(f"Backbone feature dimension: {embed_dim}")

    base_clf = nn.Linear(embed_dim, 2).to(device)
    base_clf.load_state_dict(model.model.fc.state_dict())
    base_init_state = {k: v.detach().clone() for k, v in base_clf.state_dict().items()}
    theta_0 = {"weight": base_clf.weight.detach().clone(), "bias": base_clf.bias.detach().clone()}

    val_avg, val_wga, val_groups = eval_worst_group_accuracy(
        val_feats, val_labels, val_envs, theta_0, device, n_groups=eval_n_groups
    )
    print(
        f"ERM baseline (validation): avg_acc={val_avg:.2f}% "
        f"worst_group_acc={val_wga:.2f}% groups={format_group_values(val_groups)}"
    )

    n_support = int(args.support_size_per_class)
    idx0 = class0_idx[torch.randperm(len(class0_idx), generator=g)[:n_support]]
    idx1 = class1_idx[torch.randperm(len(class1_idx), generator=g)[:n_support]]
    init_indices = [int(i) for i in torch.cat([idx0, idx1], dim=0).tolist()]

    support_init = compute_features_for_indices(
        train_dataset, init_indices, backbone, device, int(args.batch_size)
    )
    support_emb = nn.Parameter(support_init.clone())
    support_labels = torch.cat(
        [torch.zeros((n_support,), dtype=torch.long), torch.ones((n_support,), dtype=torch.long)],
        dim=0,
    ).to(device)

    base_clf.load_state_dict(base_init_state)
    theta_0 = {"weight": base_clf.weight.detach().clone(), "bias": base_clf.bias.detach().clone()}
    head_opt = torch.optim.Adam(base_clf.parameters(), lr=float(args.inner_lr))
    support_opt = torch.optim.Adam([support_emb], lr=float(args.outer_lr))

    best_worst_group = -1.0
    best_state = None

    for epoch in range(int(args.epochs)):
        hard_x, hard_y = select_hard_set(
            val_feats, val_labels, theta_0, int(args.K_hard), device
        )

        last_outer_loss = None
        for _ in range(int(args.KH)):
            theta = inner_loop_adapt_head(
                support_emb, support_labels, theta_0, int(args.T), float(args.inner_lr)
            )
            logits_outer = forward_linear(hard_x, theta)
            loss_outer = nn.CrossEntropyLoss()(logits_outer, hard_y)

            support_opt.zero_grad(set_to_none=True)
            loss_outer.backward()
            torch.nn.utils.clip_grad_norm_([support_emb], max_norm=10.0)
            support_opt.step()
            last_outer_loss = float(loss_outer.item())

        for _ in range(int(args.T)):
            head_opt.zero_grad(set_to_none=True)
            logits = base_clf(support_emb.detach())
            loss_head = nn.CrossEntropyLoss()(logits, support_labels)
            loss_head.backward()
            head_opt.step()

        theta_0 = {"weight": base_clf.weight.detach().clone(), "bias": base_clf.bias.detach().clone()}

        with torch.no_grad():
            val_avg, val_wga, val_groups = eval_worst_group_accuracy(
                val_feats, val_labels, val_envs, theta_0, device, n_groups=eval_n_groups
            )
            print(
                f"[epoch {epoch:03d}] outer_loss={last_outer_loss:.4f} "
                f"val_avg_acc={val_avg:.2f}% val_worst_group_acc={val_wga:.2f}% "
                f"groups={format_group_values(val_groups)}"
            )
            if val_wga >= best_worst_group:
                best_worst_group = val_wga
                best_state = {
                    "support_emb": support_emb.detach().cpu(),
                    "support_init": support_init.detach().cpu(),
                    "support_labels": support_labels.detach().cpu(),
                    "theta": {k: v.detach().cpu() for k, v in theta_0.items()},
                    "epoch": int(epoch),
                    "val_avg_acc": float(val_avg),
                    "val_worst_group_acc": float(val_wga),
                    "group_acc": val_groups,
                    "init_indices": init_indices,
                }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "support_emb": support_emb.detach().cpu(),
            "support_labels": support_labels.detach().cpu(),
            "theta": theta_0,
            "dataset": args.dataset,
        },
        save_path,
    )
    print(f"Saved final checkpoint: {save_path}")

    if best_state is not None:
        best_path = Path(args.best_save_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, best_path)
        print(
            f"Saved best checkpoint: {best_path} "
            f"(val_worst_group_acc={best_state['val_worst_group_acc']:.2f}% at epoch {best_state['epoch']})"
        )

        if args.save_repr_checkpoints:
            repr_dir = Path(args.repr_checkpoint_root) / args.dataset
            repr_dir.mkdir(parents=True, exist_ok=True)
            repr_path = repr_dir / "best_support_embeddings.pt"
            torch.save(
                {
                    "dataset": args.dataset,
                    "best_epoch": int(best_state["epoch"]),
                    "init_indices": best_state["init_indices"],
                    "support_labels": best_state["support_labels"],
                    "support_before": best_state["support_init"],
                    "support_after": best_state["support_emb"],
                    "val_worst_group_acc": float(best_state["val_worst_group_acc"]),
                },
                repr_path,
            )
            print(f"Saved representation checkpoint for visualization: {repr_path}")

        best_theta = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in best_state["theta"].items()
        }
        with torch.no_grad():
            test_avg, test_wga, test_groups = eval_worst_group_accuracy(
                test_feats, test_labels, test_envs, best_theta, device, n_groups=eval_n_groups
            )
            print(
                f"Best model (test): avg_acc={test_avg:.2f}% "
                f"worst_group_acc={test_wga:.2f}% groups={format_group_values(test_groups)}"
            )


if __name__ == "__main__":
    main()
