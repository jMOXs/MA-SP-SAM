#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ma_sp_sam.data.dataset_tem import DatasetTEM
from ma_sp_sam.losses.self_prompt_losses import SelfPromptLoss
from ma_sp_sam.models import SelfPromptGenerator
from ma_sp_sam.prompts.proposal_generator import ProposalGenerator
from ma_sp_sam.utils.io import load_yaml


TENSOR_KEYS = ("image", "semantic", "center_heatmap", "boundary_inner", "boundary_outer", "distance_map")


class SyntheticSelfPromptDataset(Dataset):
    def __init__(self, *, samples: int = 4, height: int = 32, width: int = 32) -> None:
        self.samples = samples
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        h, w = self.height, self.width
        image = torch.zeros(1, h, w, dtype=torch.float32)
        semantic = torch.zeros(h, w, dtype=torch.long)
        y0 = 4 + (index % 3)
        x0 = 5 + (index % 4)
        y1 = min(h - 4, y0 + 12)
        x1 = min(w - 4, x0 + 12)
        semantic[y0:y1, x0:x1] = 1
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        semantic[max(y0, cy - 2) : min(y1, cy + 2), max(x0, cx - 2) : min(x1, cx + 2)] = 2
        image[0] = semantic.float() / 2.0

        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        center = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2).float() / (2.0 * 2.0**2)).unsqueeze(0)
        boundary_inner = torch.zeros(1, h, w, dtype=torch.float32)
        boundary_outer = torch.zeros(1, h, w, dtype=torch.float32)
        boundary_outer[:, y0:y1, x0] = 1
        boundary_outer[:, y0:y1, x1 - 1] = 1
        boundary_outer[:, y0, x0:x1] = 1
        boundary_outer[:, y1 - 1, x0:x1] = 1
        axon = semantic == 2
        boundary_inner[:, axon] = 1

        distance = torch.zeros(2, h, w, dtype=torch.float32)
        foreground = semantic > 0
        scale = max(float(max(y1 - y0, x1 - x0) / 2.0), 1.0)
        distance[0, foreground] = (xx[foreground].float() - cx) / scale
        distance[1, foreground] = (yy[foreground].float() - cy) / scale
        fiber_instance = torch.zeros(h, w, dtype=torch.long)
        fiber_instance[foreground] = 1
        axon_instance = torch.zeros(h, w, dtype=torch.long)
        axon_instance[semantic == 2] = 1
        return {
            "dataset": "synthetic",
            "split": "train",
            "sample_id": f"synthetic_{index:04d}",
            "image": image,
            "semantic": semantic,
            "fiber_instance": fiber_instance,
            "axon_instance": axon_instance,
            "center_heatmap": center.float(),
            "boundary_inner": boundary_inner,
            "boundary_outer": boundary_outer,
            "distance_map": distance,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MA-SP-SAM SelfPromptGenerator V1.")
    parser.add_argument("--config", default="configs/train/self_prompt.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = load_yaml(config_path)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.limit is not None:
        config["limit"] = args.limit
    if args.device is not None:
        config["device"] = args.device
    train(config)


def train(config: dict[str, Any]) -> Path:
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset, val_dataset = build_datasets(config)
    if limit := config.get("limit"):
        train_dataset = Subset(train_dataset, range(min(int(limit), len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(int(limit), len(val_dataset))))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collate_training_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        collate_fn=collate_training_batch,
    )

    model = SelfPromptGenerator(
        in_channels=int(config.get("in_channels", 1)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_blocks=int(config.get("num_blocks", 2)),
        num_classes=3,
        distance_channels=2,
    ).to(device)
    loss_weights = config.get("loss_weights", {}) or {}
    criterion = SelfPromptLoss(
        semantic_weight=float(loss_weights.get("semantic", 1.0)),
        center_weight=float(loss_weights.get("center", 1.0)),
        boundary_weight=float(loss_weights.get("boundary", 1.0)),
        distance_weight=float(loss_weights.get("distance", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    checkpoint_dir = _resolve(config.get("checkpoint_dir", "checkpoints/self_prompt"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_loss = float("inf")
    epochs = int(config.get("epochs", 1))
    summary_interval = max(1, int(config.get("proposal_summary_interval", 1)))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"epoch={epoch} train_loss={train_loss:.6f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "train_loss": train_loss,
                },
                best_path,
            )
        if epoch % summary_interval == 0:
            summary = proposal_summary(
                model,
                val_loader,
                device=device,
                center_threshold=float(config.get("center_threshold", 0.5)),
                boundary_threshold=float(config.get("boundary_threshold", 0.5)),
            )
            print(
                "proposal_summary "
                f"samples={summary['samples']} proposals={summary['proposals']} "
                f"mean_per_sample={summary['mean_per_sample']:.3f}"
            )
    return best_path


def train_one_epoch(
    model: SelfPromptGenerator,
    loader: DataLoader,
    criterion: SelfPromptLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images, targets = batch_to_device(batch, device)
        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def proposal_summary(
    model: SelfPromptGenerator,
    loader: DataLoader,
    *,
    device: torch.device,
    center_threshold: float,
    boundary_threshold: float,
) -> dict[str, float]:
    model.eval()
    generator = ProposalGenerator(
        center_threshold=center_threshold,
        boundary_threshold=boundary_threshold,
        min_area=1,
    )
    samples = 0
    proposals = 0
    for batch in loader:
        images, _ = batch_to_device(batch, device)
        outputs = model(images)
        boundary_maps = torch.cat(
            [torch.sigmoid(outputs.inner_boundary_map), torch.sigmoid(outputs.outer_boundary_map)],
            dim=1,
        )
        proposal_batch = generator.generate(
            semantic_logits=outputs.semantic_logits.detach().cpu(),
            center_heatmap=torch.sigmoid(outputs.axon_center_heatmap).detach().cpu(),
            boundary_maps=boundary_maps.detach().cpu(),
        )
        samples += len(proposal_batch.proposals)
        proposals += sum(len(item) for item in proposal_batch.proposals)
    return {
        "samples": float(samples),
        "proposals": float(proposals),
        "mean_per_sample": float(proposals / max(samples, 1)),
    }


def build_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    if config.get("synthetic", False):
        dataset = SyntheticSelfPromptDataset(
            samples=int(config.get("synthetic_samples", 4)),
            height=int(config.get("synthetic_height", 32)),
            width=int(config.get("synthetic_width", 32)),
        )
        return dataset, dataset
    manifest_path = _resolve(config["manifest_path"])
    processed_root = _resolve(config["processed_root"])
    dataset_name = config.get("dataset", "TEM1")
    return (
        DatasetTEM(
            manifest_path,
            processed_root=processed_root,
            dataset=dataset_name,
            split=config.get("train_split", "train"),
            return_tensors=True,
            center_target="gaussian",
            center_sigma=float(config.get("center_sigma", 3.0)),
        ),
        DatasetTEM(
            manifest_path,
            processed_root=processed_root,
            dataset=dataset_name,
            split=config.get("val_split", "test"),
            return_tensors=True,
            center_target="gaussian",
            center_sigma=float(config.get("center_sigma", 3.0)),
        ),
    )


def collate_training_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_h = max(int(item["image"].shape[-2]) for item in items)
    max_w = max(int(item["image"].shape[-1]) for item in items)
    batch: dict[str, Any] = {}
    for key in TENSOR_KEYS:
        values = [_pad_tensor(item[key], max_h=max_h, max_w=max_w) for item in items]
        batch[key] = torch.stack(values, dim=0)
    batch["sample_id"] = [item.get("sample_id") for item in items]
    return batch


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = batch["image"].to(device)
    targets = {key: batch[key].to(device) for key in TENSOR_KEYS if key != "image"}
    return images, targets


def _pad_tensor(tensor: torch.Tensor, *, max_h: int, max_w: int) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    pad_h = max_h - h
    pad_w = max_w - w
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
