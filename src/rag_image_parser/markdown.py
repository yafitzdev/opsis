from __future__ import annotations

import ast
import json
import re
from typing import Any

_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")
_WRAPPER_TAGS = re.compile(r"^</?(?:table|description)>\s*$", re.IGNORECASE)
_OPENING_WRAPPER = re.compile(r"^\s*<(?:table|description)>\s*", re.IGNORECASE)
_CLOSING_WRAPPER = re.compile(r"\s*</(?:table|description)>\s*$", re.IGNORECASE)


def _strip_wrappers(text: str) -> list[str]:
    unwrapped = _OPENING_WRAPPER.sub("", text.strip())
    unwrapped = _CLOSING_WRAPPER.sub("", unwrapped)
    lines = [line.rstrip() for line in unwrapped.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or _WRAPPER_TAGS.match(stripped):
            continue
        cleaned.append(line)
    return cleaned


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_delimiter_row(line: str) -> bool:
    cells = _cells(line)
    return len(cells) >= 2 and all(_DELIMITER_CELL.fullmatch(cell) for cell in cells)


def extract_markdown_table(text: str) -> str | None:
    """Extract one structurally valid Markdown table from generated text."""
    lines = _strip_wrappers(text)
    for delimiter_index, line in enumerate(lines):
        if delimiter_index == 0 or not is_delimiter_row(line):
            continue

        header = lines[delimiter_index - 1]
        expected_columns = len(_cells(line))
        if "|" not in header or len(_cells(header)) != expected_columns:
            continue

        table_lines = [header.strip(), line.strip()]
        for row in lines[delimiter_index + 1 :]:
            if "|" not in row or len(_cells(row)) != expected_columns:
                break
            table_lines.append(row.strip())

        return "\n".join(table_lines)
    return None


def _escape_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def extract_structured_table(text: str) -> str | None:
    """Convert a model-emitted JSON/Python table object into Markdown."""
    candidate = "\n".join(_strip_wrappers(text)).strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None

    parsed: Any
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(parsed, dict):
        return None
    header = parsed.get("header", parsed.get("headers"))
    rows = parsed.get("rows", parsed.get("data"))
    if not isinstance(header, list) or len(header) < 2 or not isinstance(rows, list):
        return None
    if not all(isinstance(row, list) and len(row) == len(header) for row in rows):
        return None

    header_line = "| " + " | ".join(_escape_cell(cell) for cell in header) + " |"
    delimiter = "| " + " | ".join("---" for _ in header) + " |"
    row_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_line, delimiter, *row_lines])


def canonical_table(text: str) -> tuple[tuple[str, ...], ...] | None:
    """Represent a Markdown table by cell contents, ignoring delimiter formatting."""
    table = extract_markdown_table(text)
    if table is None:
        return None
    lines = table.splitlines()
    return tuple(tuple(_cells(line)) for index, line in enumerate(lines) if index != 1)


def clean_description(text: str) -> str:
    lines = _strip_wrappers(text)
    return " ".join(part.strip() for part in lines if part.strip())
