from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ma_sp_sam.utils.io import read_jsonl, write_jsonl


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}
MASK_SUFFIXES = {
    "axonmyelin": "_seg-axonmyelin-manual.png",
    "axon": "_seg-axon-manual.png",
    "myelin": "_seg-myelin-manual.png",
    "uaxon": "_seg-uaxon-manual.png",
    "process": "_seg-process-manual.png",
    "nuclei": "_seg-nuclei-manual.png",
}


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    split: str
    sample_id: str
    image_path: Path
    axonmyelin_mask_path: Path | None
    axon_mask_path: Path | None
    myelin_mask_path: Path | None
    auxiliary_mask_paths: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "sample_id": self.sample_id,
            "image_path": str(self.image_path),
            "axonmyelin_mask_path": str(self.axonmyelin_mask_path) if self.axonmyelin_mask_path else None,
            "axon_mask_path": str(self.axon_mask_path) if self.axon_mask_path else None,
            "myelin_mask_path": str(self.myelin_mask_path) if self.myelin_mask_path else None,
            "auxiliary_mask_paths": {key: str(path) for key, path in self.auxiliary_mask_paths.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleRecord":
        def maybe_path(value: str | None) -> Path | None:
            return Path(value) if value else None

        return cls(
            dataset=data["dataset"],
            split=data["split"],
            sample_id=data["sample_id"],
            image_path=Path(data["image_path"]),
            axonmyelin_mask_path=maybe_path(data.get("axonmyelin_mask_path")),
            axon_mask_path=maybe_path(data.get("axon_mask_path")),
            myelin_mask_path=maybe_path(data.get("myelin_mask_path")),
            auxiliary_mask_paths={
                key: Path(value) for key, value in data.get("auxiliary_mask_paths", {}).items() if value
            },
        )


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and "_seg-" not in path.name


def _mask_path(split_dir: Path, sample_id: str, suffix: str) -> Path | None:
    candidate = split_dir / f"{sample_id}{suffix}"
    return candidate if candidate.exists() else None


def index_astih_splits(
    splits_root: str | Path,
    datasets: Iterable[str] = ("TEM1", "TEM2"),
    split_names: Iterable[str] = ("train", "test"),
) -> list[SampleRecord]:
    """Index ASTIH official split directories without creating new splits."""
    root = Path(splits_root)
    records: list[SampleRecord] = []

    for dataset in datasets:
        for split in split_names:
            split_dir = root / dataset / split
            if not split_dir.exists():
                continue
            for image_path in sorted(path for path in split_dir.iterdir() if path.is_file() and _is_image_file(path)):
                sample_id = image_path.stem
                aux = {}
                for aux_name in ("uaxon", "process", "nuclei"):
                    path = _mask_path(split_dir, sample_id, MASK_SUFFIXES[aux_name])
                    if path is not None:
                        aux[aux_name] = path
                records.append(
                    SampleRecord(
                        dataset=dataset,
                        split=split,
                        sample_id=sample_id,
                        image_path=image_path,
                        axonmyelin_mask_path=_mask_path(split_dir, sample_id, MASK_SUFFIXES["axonmyelin"]),
                        axon_mask_path=_mask_path(split_dir, sample_id, MASK_SUFFIXES["axon"]),
                        myelin_mask_path=_mask_path(split_dir, sample_id, MASK_SUFFIXES["myelin"]),
                        auxiliary_mask_paths=aux,
                    )
                )

    return records


def write_manifest(path: str | Path, records: list[SampleRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def read_manifest(path: str | Path) -> list[SampleRecord]:
    return [SampleRecord.from_dict(row) for row in read_jsonl(path)]
