from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    image: Path
    output: str


def _valid_serialization(output: str) -> bool:
    stripped = output.strip()
    return (
        stripped.startswith("<description>") and stripped.endswith("</description>")
    ) or (stripped.startswith("<table>") and stripped.endswith("</table>"))


def read_training_records(path: str | Path) -> list[TrainingRecord]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")

    records: list[TrainingRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                image_value = raw["image"]
                output = raw["output"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}") from exc

            image_path = (manifest.parent / image_value).resolve()
            if not image_path.is_file():
                raise ValueError(f"Missing image at line {line_number}: {image_path}")
            if not isinstance(output, str) or not _valid_serialization(output):
                raise ValueError(
                    f"Output at line {line_number} must use description or table tags"
                )
            records.append(TrainingRecord(image=image_path, output=output.strip()))

    if not records:
        raise ValueError("Training manifest contains no records")
    return records


class ImageCompletionDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: str | Path) -> None:
        self.records = read_training_records(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image) as opened:
            image = opened.convert("RGB")
        return {"image": image, "output": record.output}

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(len(self)):
            yield self[index]
