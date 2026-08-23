from __future__ import annotations

import csv
import html
import io
import re
from dataclasses import dataclass
from typing import Any

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_TITLE_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_TITLE_STOP_WORDS = {
    "about",
    "across",
    "after",
    "before",
    "during",
    "from",
    "have",
    "into",
    "more",
    "than",
    "that",
    "their",
    "this",
    "which",
    "with",
}


@dataclass(frozen=True, slots=True)
class DescriptionTarget:
    text: str
    required_terms: tuple[str, ...]
    required_relations: tuple[tuple[str, str], ...] = ()


def _plain_text(tokens: list[str]) -> str:
    value = html.unescape("".join(tokens))
    value = _TAG.sub(" ", value)
    return _SPACE.sub(" ", value).strip().replace("|", "\\|")


def pubtabnet_markdown(record: dict[str, Any]) -> tuple[str | None, str]:
    """Convert a simple PubTabNet record to GFM, returning a rejection reason."""
    table = record.get("html")
    if not isinstance(table, dict):
        return None, "missing_html"
    structure = table.get("structure", {}).get("tokens")
    cells = table.get("cells", table.get("cell"))
    if not isinstance(structure, list) or not isinstance(cells, list):
        return None, "missing_structure_or_cells"
    if any("rowspan" in token.casefold() or "colspan" in token.casefold() for token in structure):
        return None, "spanning_cells"

    row_widths: list[int] = []
    in_row = False
    width = 0
    for token in structure:
        lowered = token.casefold().strip()
        if lowered == "<tr>":
            in_row = True
            width = 0
        elif lowered == "</td>" and in_row:
            width += 1
        elif lowered == "</tr>" and in_row:
            row_widths.append(width)
            in_row = False
    if len(row_widths) < 2:
        return None, "too_few_rows"
    if len(set(row_widths)) != 1 or row_widths[0] < 2:
        return None, "non_rectangular"
    if row_widths[0] > 12 or len(row_widths) > 30:
        return None, "too_large_for_markdown"
    if sum(row_widths) != len(cells):
        return None, "cell_count_mismatch"

    values: list[str] = []
    for cell in cells:
        tokens = cell.get("tokens") if isinstance(cell, dict) else None
        if not isinstance(tokens, list):
            return None, "invalid_cell"
        value = _plain_text(tokens)
        if len(value) > 160 or "\n" in value:
            return None, "long_cell"
        values.append(value)

    columns = row_widths[0]
    rows = [values[index : index + columns] for index in range(0, len(values), columns)]
    if not any(rows[0]):
        return None, "empty_header"
    rendered = ["| " + " | ".join(row) + " |" for row in rows]
    rendered.insert(1, "| " + " | ".join("---" for _ in range(columns)) + " |")
    return "\n".join(rendered), "accepted"


def _number(value: str) -> float | None:
    normalized = value.strip().replace("−", "-").replace(",", "")
    normalized = normalized.removeprefix("$").removesuffix("%").strip()
    match = _NUMBER.fullmatch(normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _trim_words(text: str, maximum: int = 100) -> str:
    words = text.split()
    if len(words) <= maximum:
        return text
    return " ".join(words[:maximum]).rstrip(" ,;:") + "."


def _title_terms(title: str) -> list[str]:
    words = [
        word.strip("'-")
        for word in _TITLE_WORD.findall(title)
        if len(word.strip("'-")) >= 4
        and word.casefold().strip("'-") not in _TITLE_STOP_WORDS
    ]
    return list(dict.fromkeys(words))[:5]


def chart_description(annotation: dict[str, Any], csv_text: str) -> DescriptionTarget | None:
    info = annotation.get("general_figure_info", {})
    title = str(info.get("title", {}).get("text", "")).strip()
    chart_type = str(annotation.get("type", "chart")).replace("_", " ").strip()
    rows = list(csv.reader(io.StringIO(csv_text)))
    rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if not title or len(rows) < 2 or len(rows[0]) < 2:
        return None

    models = annotation.get("models", [])
    if chart_type == "pie" and isinstance(models, list):
        model_rows = [
            [
                str(model.get("text_label") or model.get("name") or "").strip(" ,"),
                str(model.get("value", "")).strip(),
            ]
            for model in models
            if isinstance(model, dict)
        ]
        if len(model_rows) >= 2 and all(all(cell for cell in row) for row in model_rows):
            rows = [["Entity", "Value"], *model_rows]

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None

    if isinstance(models, list) and models and chart_type != "pie":
        first_model = models[0] if isinstance(models[0], dict) else {}
        model_categories = first_model.get("x", [])
        if isinstance(model_categories, list):
            if len(model_categories) == len(rows) - 1:
                for row, label in zip(rows[1:], model_categories, strict=True):
                    row[0] = str(label).strip()
            elif len(rows) == 2 and len(model_categories) == width - 1:
                rows[0][1:] = [str(label).strip() for label in model_categories]
        model_names = [
            str(model.get("name", "")).strip(" ,")
            for model in models
            if isinstance(model, dict)
        ]
        if len(model_names) == width - 1 and all(model_names):
            rows[0][1:] = model_names
        elif len(rows) == 2 and len(model_names) == 1 and model_names[0]:
            rows[1][0] = model_names[0]
    if any("..." in cell or "�" in cell for row in rows for cell in row):
        return None

    x_name = rows[0][0] or "category"
    data_rows = rows[1:]
    value_cells = [cell for row in data_rows for cell in row[1:]]
    missing = sum(cell.casefold() in {"", "nan", "n/a", "na"} for cell in value_cells)
    if value_cells and missing / len(value_cells) > 0.25:
        return None

    if len(data_rows) == 1 and width > 3:
        categories = rows[0][1:]
        series_data = [
            (
                data_rows[0][0] or "series",
                [
                    (category, raw, _number(raw))
                    for category, raw in zip(categories, data_rows[0][1:], strict=True)
                ],
            )
        ]
    else:
        categories = [row[0] for row in data_rows if row[0]]
        series_names = [
            name or f"series {index}" for index, name in enumerate(rows[0][1:], 1)
        ]
        series_data = [
            (
                series,
                [
                    (row[0], row[column], _number(row[column]))
                    for row in data_rows
                    if row[0]
                ],
            )
            for column, series in enumerate(series_names, start=1)
        ]
    if not categories:
        return None

    facts: list[str] = []
    series_names = [series for series, _ in series_data]
    generic_labels = {"entity", "value", "category", "x", "y", "series"}
    required = [
        term
        for term in [*_title_terms(title), x_name, *series_names[:3]]
        if term.casefold() not in generic_labels
    ]
    for series, points in series_data:
        numeric = [point for point in points if point[2] is not None]
        if len(numeric) < 2:
            continue
        lowest = min(numeric, key=lambda item: item[2] if item[2] is not None else 0)
        highest = max(numeric, key=lambda item: item[2] if item[2] is not None else 0)
        if len(series_data) == 1:
            facts.append(
                f"The minimum is {lowest[1]} at {lowest[0]} and the maximum is "
                f"{highest[1]} at {highest[0]}."
            )
        else:
            facts.append(
                f"For {series}, the range is {lowest[1]} at {lowest[0]} to "
                f"{highest[1]} at {highest[0]}."
            )
        required.extend([lowest[0], lowest[1], highest[0], highest[1]])
        if len(facts) == 3:
            break

    if not facts:
        return None
    meaningful_series = [
        series for series in series_names if series.casefold() not in generic_labels
    ]
    if meaningful_series:
        comparison = f"compares {x_name} across {', '.join(meaningful_series[:5])}"
    else:
        comparison = "shows values by category"
    description = (
        f'A {chart_type} chart titled "{title}" {comparison}. ' + " ".join(facts)
    ).strip()
    unique_required = tuple(dict.fromkeys(term for term in required if term))
    return DescriptionTarget(_trim_words(description), unique_required[:10])


def ai2d_description(annotation: dict[str, Any], category: str) -> DescriptionTarget | None:
    texts = annotation.get("text", {})
    relationships = annotation.get("relationships", {})
    if not isinstance(texts, dict) or not isinstance(relationships, dict):
        return None

    text_values = {
        key: _SPACE.sub(" ", str(value.get("value", ""))).strip()
        for key, value in texts.items()
        if isinstance(value, dict) and str(value.get("value", "")).strip()
    }
    object_labels: dict[str, str] = {}
    for relation in relationships.values():
        if not isinstance(relation, dict) or relation.get("category") != "intraObjectLabel":
            continue
        origin = str(relation.get("origin", ""))
        destination = str(relation.get("destination", ""))
        if origin in text_values:
            object_labels[destination] = text_values[origin]
        elif destination in text_values:
            object_labels[origin] = text_values[destination]

    directed_links: list[tuple[str, str]] = []
    undirected_links: list[tuple[str, str]] = []
    for relation in relationships.values():
        if not isinstance(relation, dict) or relation.get("category") != "interObjectLinkage":
            continue
        origin = object_labels.get(str(relation.get("origin", "")))
        destination = object_labels.get(str(relation.get("destination", "")))
        if origin and destination and origin != destination:
            target = directed_links if relation.get("hasDirectionality") else undirected_links
            target.append((origin, destination))
    directed_links = list(dict.fromkeys(directed_links))
    undirected_links = list(dict.fromkeys(undirected_links))
    labels = list(dict.fromkeys(text_values.values()))
    if len(labels) < 2 or not (directed_links or undirected_links):
        return None

    category_text = re.sub(r"(?<!^)(?=[A-Z])", " ", category).casefold()
    clauses = []
    if directed_links:
        relation_text = "; ".join(
            f"{source} leads to {target}" for source, target in directed_links[:6]
        )
        clauses.append(f"Directed flow: {relation_text}.")
    if undirected_links:
        connection_text = "; ".join(
            f"{source} connects with {target}" for source, target in undirected_links[:6]
        )
        clauses.append(f"Connections: {connection_text}.")
    description = (
        f"A {category_text} diagram containing the labels {', '.join(labels[:8])}. "
        + " ".join(clauses)
    )
    terms = tuple(labels[:8])
    return DescriptionTarget(_trim_words(description), terms, tuple(directed_links[:6]))
