import json
from pathlib import Path

from PIL import Image

from rag_image_parser.benchmark import run_benchmark
from rag_image_parser.parser import ImageParser


class SequenceBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    @property
    def model_id(self) -> str:
        return "fake/sequence"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        return next(self.outputs)


def _manifest(tmp_path: Path, record: dict[str, object]) -> Path:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "image.png")
    manifest = tmp_path / "benchmark.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_relationship_metric_rejects_reversed_flow(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        {
            "image": "image.png",
            "expected_kind": "description",
            "required_terms": ["PDF parser", "Image model", "Vector store"],
            "required_relations": [
                {"source": "PDF parser", "target": "Image model"},
                {"source": "Image model", "target": "Vector store"},
            ],
        },
    )
    parser = ImageParser(
        SequenceBackend(["Image model connects to Vector store, then PDF parser."])
    )

    report = run_benchmark(parser, manifest)

    case = report["cases"][0]
    assert case["required_term_recall"] == 1.0
    assert case["required_relation_recall"] == 0.5
    assert case["semantic_requirements_met"] is False
    assert report["summary"]["semantic_requirement_accuracy"] == 0.0


def test_semantic_metric_accepts_directional_flow_and_absent_forbidden_term(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        {
            "image": "image.png",
            "expected_kind": "description",
            "required_terms": ["PDF parser", "Vector store"],
            "required_relations": [{"source": "PDF parser", "target": "Vector store"}],
            "forbidden_terms": ["cloud API"],
        },
    )
    parser = ImageParser(SequenceBackend(["PDF parser sends extracted images to Vector store."]))

    case = run_benchmark(parser, manifest)["cases"][0]

    assert case["semantic_requirements_met"] is True
    assert case["required_relation_recall"] == 1.0
    assert case["forbidden_terms_absent"] is True


def test_semantic_metric_normalizes_numeric_thousands_separator(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        {
            "image": "image.png",
            "expected_kind": "description",
            "required_terms": ["3410"],
        },
    )
    parser = ImageParser(SequenceBackend(["The chart shows 3,410 admissions."]))

    case = run_benchmark(parser, manifest)["cases"][0]

    assert case["required_term_recall"] == 1.0


def test_table_metrics_score_aligned_cells_and_shape(tmp_path: Path) -> None:
    expected = "| Name | Score |\n| --- | ---: |\n| Alpha | 10 |\n| Beta | 20 |"
    predicted = "| Name | Score |\n| --- | --- |\n| Alpha | 10 |\n| Beta | 21 |"
    manifest = _manifest(
        tmp_path,
        {
            "image": "image.png",
            "expected_kind": "table",
            "expected_text": expected,
        },
    )
    parser = ImageParser(SequenceBackend([predicted]))

    report = run_benchmark(parser, manifest)
    case = report["cases"][0]

    assert case["table_shape_correct"] is True
    assert case["table_cell_precision"] == 5 / 6
    assert case["table_cell_recall"] == 5 / 6
    assert case["table_cell_f1"] == 5 / 6
    assert report["summary"]["mean_table_cell_f1"] == 5 / 6


def test_table_metrics_zero_for_non_table_output(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        {
            "image": "image.png",
            "expected_kind": "table",
            "expected_text": "| A | B |\n| --- | --- |\n| 1 | 2 |",
        },
    )
    parser = ImageParser(SequenceBackend(["The image contains a table."]))

    case = run_benchmark(parser, manifest)["cases"][0]

    assert case["table_shape_correct"] is False
    assert case["table_cell_precision"] == 0.0
    assert case["table_cell_recall"] == 0.0
    assert case["table_cell_f1"] == 0.0
