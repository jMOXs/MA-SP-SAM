from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ma_sp_sam.utils.io import write_jsonl


@dataclass(frozen=True)
class AimSegRecord:
    dataset: str
    sample_id: str
    archive_path: Path | None
    image_member: str
    semantic_member: str | None
    instance_member: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "image_member": self.image_member,
            "semantic_member": self.semantic_member,
            "instance_member": self.instance_member,
        }


def list_archive_members(archive_path: str | Path) -> list[str]:
    result = subprocess.run(
        ["bsdtar", "-tf", str(archive_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def records_from_archive_members(archive_name: str | Path, members: Iterable[str]) -> list[AimSegRecord]:
    archive_path = Path(archive_name)
    grouped: dict[str, dict[str, str]] = {}
    dataset_name = None

    for member in members:
        parts = Path(member).parts
        if len(parts) < 3:
            continue
        dataset, folder, filename = parts[0], parts[1], parts[-1]
        dataset_name = dataset_name or dataset
        if folder not in {"Images", "GroundTruth_Semantic", "GroundTruth_Instance"}:
            continue
        if Path(filename).suffix.lower() not in {".tif", ".tiff", ".png"}:
            continue
        sample_id = Path(filename).stem
        grouped.setdefault(sample_id, {})
        if folder == "Images":
            grouped[sample_id]["image"] = member
        elif folder == "GroundTruth_Semantic":
            grouped[sample_id]["semantic"] = member
        elif folder == "GroundTruth_Instance":
            grouped[sample_id]["instance"] = member

    records = []
    for sample_id in sorted(grouped):
        paths = grouped[sample_id]
        if "image" not in paths:
            continue
        records.append(
            AimSegRecord(
                dataset=dataset_name or archive_path.stem,
                sample_id=sample_id,
                archive_path=archive_path,
                image_member=paths["image"],
                semantic_member=paths.get("semantic"),
                instance_member=paths.get("instance"),
            )
        )
    return records


def index_aimseg_archives(aimseg_root: str | Path) -> list[AimSegRecord]:
    root = Path(aimseg_root)
    records: list[AimSegRecord] = []
    for archive_path in sorted(root.glob("*.rar")):
        if archive_path.name == "Classifiers_v1.rar":
            continue
        members = list_archive_members(archive_path)
        records.extend(records_from_archive_members(archive_path, members))
    return records


def write_aimseg_manifest(path: str | Path, records: list[AimSegRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])
