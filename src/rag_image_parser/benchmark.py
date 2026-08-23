from __future__ import annotations

import json
import re
import statistics
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from rag_image_parser.markdown import canonical_table
from rag_image_parser.parser import ImageParser
from rag_image_parser.types import OutputKind


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    image: str
    expected_kind: str
    predicted_kind: str
    output_text: str
    kind_correct: bool
    exact_text_match: bool | None
    table_shape_correct: bool | None
    table_cell_precision: float | None
    table_cell_recall: float | None
    table_cell_f1: float | None
    semantic_requirements_met: bool | None
    required_term_recall: float | None
    required_relation_recall: float | None
    forbidden_terms_absent: bool | None
    latency_seconds: float
    output_words: int


class _PeakRssSampler:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self.start_rss = self._process.memory_info().rss
        self.peak_rss = self.start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
        self._stop.set()
        self._thread.join()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())


def _texts_match(predicted: str, expected: str, expected_kind: str) -> bool:
    if expected_kind == OutputKind.TABLE.value:
        predicted_table = canonical_table(predicted)
        expected_table = canonical_table(expected)
        return predicted_table is not None and predicted_table == expected_table
    return _normalize_text(predicted) == _normalize_text(expected)


def _normalize_cell(text: str) -> str:
    return " ".join(text.casefold().split())


def _table_metrics(
    predicted: str,
    expected: str,
) -> tuple[bool, float, float, float]:
    predicted_table = canonical_table(predicted)
    expected_table = canonical_table(expected)
    if expected_table is None:
        raise ValueError("Expected table text is not structurally valid Markdown")
    if predicted_table is None:
        return False, 0.0, 0.0, 0.0

    shape_correct = [len(row) for row in predicted_table] == [len(row) for row in expected_table]
    matched = 0
    for predicted_row, expected_row in zip(predicted_table, expected_table, strict=False):
        matched += sum(
            _normalize_cell(predicted_cell) == _normalize_cell(expected_cell)
            for predicted_cell, expected_cell in zip(
                predicted_row,
                expected_row,
                strict=False,
            )
        )
    predicted_cells = sum(len(row) for row in predicted_table)
    expected_cells = sum(len(row) for row in expected_table)
    precision = matched / predicted_cells if predicted_cells else 0.0
    recall = matched / expected_cells if expected_cells else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return shape_correct, precision, recall, f1


def _search_text(text: str) -> str:
    searchable = text.casefold().replace("’", "'")
    searchable = re.sub(r"(?<=\d),(?=\d{3}\b)", "", searchable)
    searchable = re.sub(r"[^\w.%+'-]+", " ", searchable)
    return " ".join(searchable.split())


def _string_list(raw: dict[str, Any], field: str, line_number: int) -> list[str]:
    value = raw.get(field, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings at line {line_number}")
    return value


def _relations(raw: dict[str, Any], line_number: int) -> list[tuple[str, str]]:
    value = raw.get("required_relations", [])
    if not isinstance(value, list):
        raise ValueError(f"required_relations must be a list at line {line_number}")
    relations: list[tuple[str, str]] = []
    for relation in value:
        if not isinstance(relation, dict):
            raise ValueError(f"required_relations entries must be objects at line {line_number}")
        source = relation.get("source")
        target = relation.get("target")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise ValueError(
                f"required_relations entries need non-empty source and target at line {line_number}"
            )
        relations.append((source, target))
    return relations


def _semantic_metrics(
    text: str,
    required_terms: list[str],
    required_relations: list[tuple[str, str]],
    forbidden_terms: list[str],
) -> tuple[bool | None, float | None, float | None, bool | None]:
    searchable = _search_text(text)
    term_recall = (
        sum(_search_text(term) in searchable for term in required_terms) / len(required_terms)
        if required_terms
        else None
    )
    relation_recall = None
    if required_relations:
        matched = 0
        for source, target in required_relations:
            source_text = _search_text(source)
            target_text = _search_text(target)
            source_position = searchable.find(source_text)
            target_position = searchable.find(target_text, source_position + len(source_text))
            matched += source_position >= 0 and target_position >= 0
        relation_recall = matched / len(required_relations)
    forbidden_absent = (
        not any(_search_text(term) in searchable for term in forbidden_terms)
        if forbidden_terms
        else None
    )
    configured = bool(required_terms or required_relations or forbidden_terms)
    requirements_met = (
        (term_recall is None or term_recall == 1.0)
        and (relation_recall is None or relation_recall == 1.0)
        and (forbidden_absent is None or forbidden_absent)
        if configured
        else None
    )
    return requirements_met, term_recall, relation_recall, forbidden_absent


def run_benchmark(
    parser: ImageParser,
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    cases: list[BenchmarkCase] = []
    memory = _PeakRssSampler()
    memory.start()

    try:
        parser.prepare()
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                expected_kind = raw["expected_kind"]
                if expected_kind not in {kind.value for kind in OutputKind}:
                    raise ValueError(f"Invalid expected_kind at line {line_number}")
                image = (manifest.parent / raw["image"]).resolve()
                result = parser.parse(image)
                expected_text = raw.get("expected_text")
                exact = (
                    _texts_match(result.text, expected_text, expected_kind)
                    if isinstance(expected_text, str)
                    else None
                )
                table_metrics = (
                    _table_metrics(result.text, expected_text)
                    if expected_kind == OutputKind.TABLE.value and isinstance(expected_text, str)
                    else (None, None, None, None)
                )
                required_terms = _string_list(raw, "required_terms", line_number)
                forbidden_terms = _string_list(raw, "forbidden_terms", line_number)
                required_relations = _relations(raw, line_number)
                semantic = _semantic_metrics(
                    result.text,
                    required_terms,
                    required_relations,
                    forbidden_terms,
                )
                cases.append(
                    BenchmarkCase(
                        image=str(image),
                        expected_kind=expected_kind,
                        predicted_kind=result.kind.value,
                        output_text=result.text,
                        kind_correct=result.kind.value == expected_kind,
                        exact_text_match=exact,
                        table_shape_correct=table_metrics[0],
                        table_cell_precision=table_metrics[1],
                        table_cell_recall=table_metrics[2],
                        table_cell_f1=table_metrics[3],
                        semantic_requirements_met=semantic[0],
                        required_term_recall=semantic[1],
                        required_relation_recall=semantic[2],
                        forbidden_terms_absent=semantic[3],
                        latency_seconds=result.latency_seconds,
                        output_words=len(result.text.split()),
                    )
                )
                if limit is not None and len(cases) >= limit:
                    break
    finally:
        memory.stop()

    if not cases:
        raise ValueError("Benchmark manifest contains no records")

    exact_cases = [case.exact_text_match for case in cases if case.exact_text_match is not None]
    latencies = [case.latency_seconds for case in cases]
    expected_table_cases = [case for case in cases if case.expected_kind == OutputKind.TABLE.value]
    table_shape_cases = [
        case.table_shape_correct for case in cases if case.table_shape_correct is not None
    ]
    table_cell_precisions = [
        case.table_cell_precision for case in cases if case.table_cell_precision is not None
    ]
    table_cell_recalls = [
        case.table_cell_recall for case in cases if case.table_cell_recall is not None
    ]
    table_cell_f1s = [case.table_cell_f1 for case in cases if case.table_cell_f1 is not None]
    semantic_cases = [
        case.semantic_requirements_met
        for case in cases
        if case.semantic_requirements_met is not None
    ]
    term_recalls = [
        case.required_term_recall for case in cases if case.required_term_recall is not None
    ]
    relation_recalls = [
        case.required_relation_recall for case in cases if case.required_relation_recall is not None
    ]
    return {
        "summary": {
            "cases": len(cases),
            "kind_accuracy": sum(case.kind_correct for case in cases) / len(cases),
            "exact_text_accuracy": (
                sum(bool(value) for value in exact_cases) / len(exact_cases)
                if exact_cases
                else None
            ),
            "valid_table_rate": (
                sum(case.predicted_kind == OutputKind.TABLE.value for case in expected_table_cases)
                / len(expected_table_cases)
                if expected_table_cases
                else None
            ),
            "table_shape_accuracy": (
                sum(bool(value) for value in table_shape_cases) / len(table_shape_cases)
                if table_shape_cases
                else None
            ),
            "mean_table_cell_precision": (
                statistics.fmean(table_cell_precisions) if table_cell_precisions else None
            ),
            "mean_table_cell_recall": (
                statistics.fmean(table_cell_recalls) if table_cell_recalls else None
            ),
            "mean_table_cell_f1": (statistics.fmean(table_cell_f1s) if table_cell_f1s else None),
            "semantic_requirement_accuracy": (
                sum(bool(value) for value in semantic_cases) / len(semantic_cases)
                if semantic_cases
                else None
            ),
            "mean_required_term_recall": (statistics.fmean(term_recalls) if term_recalls else None),
            "mean_required_relation_recall": (
                statistics.fmean(relation_recalls) if relation_recalls else None
            ),
            "mean_latency_seconds": statistics.fmean(latencies),
            "median_latency_seconds": statistics.median(latencies),
            "p95_latency_seconds": _percentile(latencies, 0.95),
            "peak_rss_megabytes": memory.peak_rss / (1024 * 1024),
            "peak_rss_increase_megabytes": (memory.peak_rss - memory.start_rss) / (1024 * 1024),
        },
        "cases": [asdict(case) for case in cases],
    }
