import json
from pathlib import Path

from PIL import Image

from rag_image_parser.dataset import ImageCompletionDataset, read_training_records


def test_reads_relative_training_image(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": "photo.png",
                "output": "<description>A blank white square.</description>",
            }
        ),
        encoding="utf-8",
    )

    records = read_training_records(manifest)
    dataset = ImageCompletionDataset(manifest)

    assert records[0].image == image_path.resolve()
    assert len(dataset) == 1
    assert dataset[0]["image"].mode == "RGB"


def test_rejects_unwrapped_output(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"image": "photo.png", "output": "A blank square."}),
        encoding="utf-8",
    )

    try:
        read_training_records(manifest)
    except ValueError as exc:
        assert "description or table tags" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
