import json
from pathlib import Path

import pytest
from PIL import Image

from rag_image_parser.data_builder import build_dataset, discover_labeled_images
from rag_image_parser.dataset import read_training_records


def _image(path: Path) -> None:
    Image.new("RGB", (12, 12), "white").save(path)


def test_builds_stratified_deterministic_manifests(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "manifests"
    source.mkdir()
    for index in range(4):
        image = source / f"description-{index}.png"
        _image(image)
        image.with_suffix(".txt").write_text(f"Visible subject {index}.", encoding="utf-8")
    for index in range(4):
        image = source / f"table-{index}.png"
        _image(image)
        image.with_suffix(".md").write_text(
            f"| Item | Value |\n| --- | --- |\n| A | {index} |", encoding="utf-8"
        )

    counts = build_dataset(source, output, validation_fraction=0.25, seed=9)
    first_validation = (output / "validation.jsonl").read_text(encoding="utf-8")
    repeated_counts = build_dataset(source, output, validation_fraction=0.25, seed=9)

    assert counts == (6, 2)
    assert repeated_counts == counts
    assert (output / "validation.jsonl").read_text(encoding="utf-8") == first_validation
    validation_rows = [json.loads(line) for line in first_validation.splitlines()]
    assert {row["kind"] for row in validation_rows} == {"description", "table"}
    assert len(read_training_records(output / "train.jsonl")) == 6
    assert len(read_training_records(output / "validation.jsonl")) == 2


def test_rejects_unlabeled_image(tmp_path: Path) -> None:
    _image(tmp_path / "orphan.png")

    with pytest.raises(ValueError, match="no .txt or .md sidecar"):
        discover_labeled_images(tmp_path)


def test_rejects_table_sidecar_with_extra_prose(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    _image(image)
    image.with_suffix(".md").write_text(
        "Caption\n\n| A | B |\n| --- | --- |\n| 1 | 2 |", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="only one valid Markdown table"):
        discover_labeled_images(tmp_path)
