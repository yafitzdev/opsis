import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.curate_training_g2 import (
    TRAIN_PDF_CONTENT,
    TRAIN_TOP_LEVEL,
    RenderJob,
    _digest,
    allocate,
    deduplicate_image_splits,
    source_identity,
)


def _item(name: str) -> dict[str, str]:
    return {
        "source": "test",
        "source_split": "train",
        "filename": name,
    }


def test_g2_distribution_is_exactly_50k() -> None:
    assert sum(TRAIN_TOP_LEVEL.values()) == 50_000
    assert sum(TRAIN_PDF_CONTENT.values()) == TRAIN_TOP_LEVEL["pdf"]


def test_allocate_cycles_deterministically() -> None:
    items = [_item("a.png"), _item("b.png")]
    first = allocate(items, 5)
    second = allocate(items, 5)

    assert first == second
    assert sorted(variant for _, variant in first) == [0, 0, 1, 1, 2]


def test_render_digest_includes_domain() -> None:
    identity = source_identity(_item("a.png"))
    clean = RenderJob(Path("a"), Path("b"), "clean", "image", 0, identity)
    pdf = RenderJob(Path("a"), Path("c"), "pdf", "image", 0, identity)

    assert _digest(clean) != _digest(pdf)


def test_image_deduplication_blocks_cross_split_leakage(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    different = tmp_path / "different.png"
    first.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    different.write_bytes(b"different")
    train = [{**_item("a.png"), "image": first}]
    evaluation = [
        {**_item("b.png"), "image": duplicate},
        {**_item("c.png"), "image": different},
    ]

    unique_train, unique_eval, report = deduplicate_image_splits(train, evaluation)

    assert unique_train == train
    assert unique_eval == [evaluation[1]]
    assert report["cross_split_images_removed"] == 1
