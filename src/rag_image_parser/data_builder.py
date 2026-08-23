from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from rag_image_parser.markdown import extract_markdown_table

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class LabeledImage:
    image: Path
    output: str
    kind: str


def discover_labeled_images(input_dir: str | Path) -> list[LabeledImage]:
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    records: list[LabeledImage] = []
    for image_path in sorted(
        path for path in root.rglob("*") if path.suffix.casefold() in IMAGE_EXTENSIONS
    ):
        description_path = image_path.with_suffix(".txt")
        table_path = image_path.with_suffix(".md")
        sidecars = [path for path in (description_path, table_path) if path.is_file()]
        relative = image_path.relative_to(root)
        if not sidecars:
            raise ValueError(f"Image has no .txt or .md sidecar: {relative}")
        if len(sidecars) > 1:
            raise ValueError(f"Image has both description and table sidecars: {relative}")

        try:
            with Image.open(image_path) as opened:
                opened.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Unreadable image: {relative}") from exc

        sidecar = sidecars[0]
        label = sidecar.read_text(encoding="utf-8").strip()
        if sidecar.suffix.casefold() == ".md":
            table = extract_markdown_table(label)
            if table is None or table.strip() != label:
                raise ValueError(
                    f"Table sidecar must contain only one valid Markdown table: {sidecar}"
                )
            output = f"<table>\n{table}\n</table>"
            kind = "table"
        else:
            if not label:
                raise ValueError(f"Description sidecar is empty: {sidecar}")
            if "<description>" in label or "<table>" in label:
                raise ValueError(f"Sidecars must not include training tags: {sidecar}")
            output = f"<description>{label}</description>"
            kind = "description"
        records.append(LabeledImage(image=image_path.resolve(), output=output, kind=kind))

    if not records:
        raise ValueError(f"No supported images found in {root}")
    return records


def split_records(
    records: list[LabeledImage],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[LabeledImage], list[LabeledImage]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    train: list[LabeledImage] = []
    validation: list[LabeledImage] = []
    for kind in ("description", "table"):
        group = [record for record in records if record.kind == kind]
        ranked = sorted(
            group,
            key=lambda record: hashlib.sha256(
                f"{seed}:{record.image.as_posix()}".encode()
            ).digest(),
        )
        validation_count = round(len(ranked) * validation_fraction)
        if validation_fraction > 0 and len(ranked) > 1:
            validation_count = min(max(validation_count, 1), len(ranked) - 1)
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])
    return sorted(train, key=lambda record: str(record.image)), sorted(
        validation, key=lambda record: str(record.image)
    )


def _write_manifest(path: Path, records: list[LabeledImage]) -> None:
    lines = []
    for record in records:
        relative_image = Path(os.path.relpath(record.image, path.parent)).as_posix()
        lines.append(
            json.dumps(
                {"image": relative_image, "output": record.output, "kind": record.kind},
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> tuple[int, int]:
    records = discover_labeled_images(input_dir)
    train, validation = split_records(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_manifest(destination / "train.jsonl", train)
    _write_manifest(destination / "validation.jsonl", validation)
    return len(train), len(validation)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-image-build-dataset",
        description="Build training manifests from image + sidecar label pairs",
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train_count, validation_count = build_dataset(
        args.input_dir,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(f"Built {train_count} train and {validation_count} validation examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
